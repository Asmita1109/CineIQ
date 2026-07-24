"""Train the CineIQ NCF recommender.

The actual training loop lives in train_model() so it can be reused verbatim
by sagemaker_train.py -- only the data/model paths and hyperparameter source
differ between a local run and a SageMaker training job.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import ndcg_score
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from model import NCF, CineIQDataset, CineIQIterableDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = MODELS_DIR / "figures"

EMBEDDING_DIM = 32
HIDDEN_LAYERS = (128, 64, 32)
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
NDCG_K = 10
PALETTE = "mako"


def compute_ndcg_at_k(user_idx, y_true, y_pred, k=NDCG_K):
    """Per-user NDCG@k, ranking each user's own known items by predicted
    rating and scoring against the ideal (actual-rating-sorted) order."""
    order = np.argsort(user_idx, kind="stable")
    user_idx_sorted = user_idx[order]
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]

    unique_users, counts = np.unique(user_idx_sorted, return_counts=True)
    group_size = counts[0]

    if np.all(counts == group_size):
        # fast path: fixed items/user (true for the leave-last-5-out rec split)
        n_users = len(unique_users)
        true_matrix = y_true_sorted.reshape(n_users, group_size)
        pred_matrix = y_pred_sorted.reshape(n_users, group_size)
        return float(ndcg_score(true_matrix, pred_matrix, k=k))

    # fallback for ragged group sizes
    scores = []
    start = 0
    for c in counts:
        if c >= 2:
            t = y_true_sorted[start:start + c].reshape(1, -1)
            p = y_pred_sorted[start:start + c].reshape(1, -1)
            scores.append(ndcg_score(t, p, k=k))
        start += c
    return float(np.mean(scores)) if scores else float("nan")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred, all_users = [], [], []
    for user_idx, movie_idx, genome, rating in loader:
        user_idx, movie_idx, genome = user_idx.to(device), movie_idx.to(device), genome.to(device)
        preds = model(user_idx, movie_idx, genome).cpu().numpy()
        all_true.append(rating.numpy())
        all_pred.append(preds)
        all_users.append(user_idx.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    users = np.concatenate(all_users)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ndcg = compute_ndcg_at_k(users, y_true, y_pred, k=NDCG_K)
    return rmse, ndcg


def plot_training_curve(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    train_rmse = [x**0.5 for x in history["train_loss"]]
    colors = sns.color_palette(PALETTE, 5)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, train_rmse, label="train RMSE", color=colors[3])
    ax1.plot(epochs, history["val_rmse"], label="val RMSE", color=colors[1])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("RMSE")
    ax1.set_title("Recommender Training Curve")

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_ndcg"], label=f"val NDCG@{NDCG_K}", color="firebrick", linestyle="--")
    ax2.set_ylabel(f"NDCG@{NDCG_K}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def train_model(
    train_path,
    val_path,
    user_features_path,
    movie_features_path,
    model_output_path=None,
    figures_dir=None,
    embedding_dim=EMBEDDING_DIM,
    hidden_layers=HIDDEN_LAYERS,
    lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    max_epochs=MAX_EPOCHS,
    patience=EARLY_STOPPING_PATIENCE,
    save_model=True,
    max_train_rows=None,
    dataset_class=CineIQDataset,
    chunk_size=None,
):
    """dataset_class/chunk_size let a caller swap in CineIQIterableDataset
    (streamed, chunked pyarrow reads -- for memory-constrained training
    instances) instead of the default CineIQDataset (loads everything into
    memory up front) without touching the model/loss/optimizer/eval logic
    below, which is identical either way."""
    if save_model and model_output_path is None:
        raise ValueError("model_output_path is required when save_model=True")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    streaming = issubclass(dataset_class, IterableDataset)
    extra_kwargs = {}
    if streaming:
        extra_kwargs["batch_size"] = batch_size
        if chunk_size is not None:
            extra_kwargs["chunk_size"] = chunk_size
    elif max_train_rows is not None:
        extra_kwargs["max_rows"] = max_train_rows

    print(f"Loading datasets ({'streamed/chunked' if streaming else 'full in-memory'})...")
    train_dataset = dataset_class(train_path, user_features_path, movie_features_path, **extra_kwargs)
    val_dataset = dataset_class(
        val_path,
        user_features_path,
        movie_features_path,
        user_id_map=train_dataset.user_id_map,
        movie_id_map=train_dataset.movie_id_map,
        **({"batch_size": batch_size, **({"chunk_size": chunk_size} if chunk_size is not None else {})} if streaming else {}),
    )
    print(f"  train: {len(train_dataset):,} interactions")
    print(f"  val:   {len(val_dataset):,} interactions")
    print(
        f"  users: {train_dataset.num_users:,}  movies: {train_dataset.num_movies:,}  "
        f"genome_dim: {train_dataset.genome_dim}"
    )

    if streaming:
        # CineIQIterableDataset already yields exactly batch_size-sized
        # batches and reads sequentially -- batch_size=None tells DataLoader
        # not to re-batch, and shuffle isn't supported for IterableDataset.
        train_loader = DataLoader(train_dataset, batch_size=None)
        val_loader = DataLoader(val_dataset, batch_size=None)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = NCF(
        num_users=train_dataset.num_users,
        num_movies=train_dataset.num_movies,
        embedding_dim=embedding_dim,
        genome_dim=train_dataset.genome_dim,
        hidden_layers=hidden_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_rmse": [], "val_ndcg": []}
    best_val_rmse = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_examples = 0
        for user_idx, movie_idx, genome, rating in train_loader:
            user_idx, movie_idx, genome, rating = (
                user_idx.to(device),
                movie_idx.to(device),
                genome.to(device),
                rating.to(device),
            )
            optimizer.zero_grad()
            preds = model(user_idx, movie_idx, genome)
            loss = criterion(preds, rating)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(rating)
            n_examples += len(rating)

        train_loss = total_loss / n_examples
        val_rmse, val_ndcg = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_rmse"].append(val_rmse)
        history["val_ndcg"].append(val_ndcg)
        print(
            f"Epoch {epoch:3d} | train MSE: {train_loss:.4f} | val RMSE: {val_rmse:.4f} | "
            f"val NDCG@{NDCG_K}: {val_ndcg:.4f}"
        )

        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            epochs_without_improvement = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "num_users": train_dataset.num_users,
                "num_movies": train_dataset.num_movies,
                "embedding_dim": embedding_dim,
                "genome_dim": train_dataset.genome_dim,
                "hidden_layers": list(hidden_layers),
                "user_id_map": train_dataset.user_id_map,
                "movie_id_map": train_dataset.movie_id_map,
                "epoch": epoch,
                "val_rmse": val_rmse,
                "val_ndcg": val_ndcg,
            }
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no val RMSE improvement for {patience} epochs)")
                break

    if save_model:
        model_output_path = Path(model_output_path)
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, model_output_path)
        print(
            f"Saved best model (epoch {best_state['epoch']}, val RMSE {best_state['val_rmse']:.4f}, "
            f"val NDCG@{NDCG_K} {best_state['val_ndcg']:.4f}) -> {model_output_path}"
        )
    else:
        print(
            f"save_model=False -- skipping checkpoint save "
            f"(best epoch {best_state['epoch']}, val RMSE {best_state['val_rmse']:.4f}, "
            f"val NDCG@{NDCG_K} {best_state['val_ndcg']:.4f})"
        )

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 150
        plot_training_curve(history, figures_dir / "recommender_loss_curve.png")

    return best_state, history


def main():
    train_model(
        train_path=FEATURES_DIR / "rec_train.parquet",
        val_path=FEATURES_DIR / "rec_val.parquet",
        user_features_path=FEATURES_DIR / "user_features.parquet",
        movie_features_path=FEATURES_DIR / "movie_features.parquet",
        model_output_path=MODELS_DIR / "recommender_model.pt",
        figures_dir=FIGURES_DIR,
    )


if __name__ == "__main__":
    main()
