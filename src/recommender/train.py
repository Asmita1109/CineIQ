"""Train the CineIQ NCF recommender with BPR pairwise ranking loss (Version 3).

Replaces the earlier pointwise approach (MSE against star ratings) with
Bayesian Personalized Ranking: for each observed positive interaction,
neg_samples movies the user has never rated are sampled, and the model is
pushed to score the positive higher than each negative via
loss = -log(sigmoid(score(positive) - score(negative))), averaged over every
(positive, negative) pair in a batch.

Validation now uses the same negative-sampling protocol as
src/recommender/evaluate.py's test-set evaluation (100 negatives per user,
rank the real held-out items among them) -- this file imports
sample_negatives/score_candidates/per_user_ranking_metrics_negsample from
evaluate.py rather than re-implementing already-validated logic, so
training-time NDCG@10 is directly comparable to the final test-set numbers.

The actual training loop lives in train_model() so it can be reused
verbatim by sagemaker_train.py -- only the data/model paths, hyperparameter
source, and data loading strategy (map-style vs. streamed) differ between a
local run and a SageMaker training job.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from evaluate import per_user_ranking_metrics_negsample, sample_negatives, score_candidates
from model import NCF, CineIQBPRDataset, CineIQBPRIterableDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = MODELS_DIR / "figures"

EMBEDDING_DIM = 64
HIDDEN_LAYERS = (128, 64, 32)
DROPOUT = 0.2
BATCH_SIZE = 1024
LEARNING_RATE = 0.0005
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7
NEG_SAMPLES = 4          # training: negatives sampled per positive interaction
VAL_NEG_SAMPLES = 100    # validation: negatives sampled per user, for ranking metrics
K = 10
PALETTE = "mako"


def load_rating_history(paths, user_ids):
    """Every movie any of user_ids has rated, per the given parquet file(s)
    -- used to exclude true positives from validation negative sampling.
    Pass a single path (e.g. rl_features.parquet, the fullest available
    interaction log, used by default for local runs) or a list of paths
    (e.g. [train_path, val_path] -- always available even on SageMaker,
    where a full unified interaction-log channel isn't set up)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    user_id_set = set(user_ids)
    frames = []
    for p in paths:
        df = pd.read_parquet(p, columns=["userId", "movieId"])
        frames.append(df[df["userId"].isin(user_id_set)])
    history = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return history.drop_duplicates()


def bpr_loss(pos_score, neg_score):
    """pos_score: (B,), neg_score: (B, neg_samples). Broadcasts pos_score
    across the negative dimension and averages the loss over every
    (positive, negative) pair in the batch."""
    diff = pos_score.unsqueeze(1) - neg_score
    return -F.logsigmoid(diff).mean()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_examples = 0
    for user_idx, pos_movie_idx, pos_genome, neg_movie_idx, neg_genome in loader:
        batch_size, neg_samples = neg_movie_idx.shape

        user_idx = user_idx.to(device)
        pos_movie_idx = pos_movie_idx.to(device)
        pos_genome = pos_genome.to(device)
        neg_movie_idx = neg_movie_idx.to(device)
        neg_genome = neg_genome.to(device)

        optimizer.zero_grad()

        pos_score = model(user_idx, pos_movie_idx, pos_genome)  # (B,)

        user_idx_expanded = user_idx.unsqueeze(1).expand(-1, neg_samples).reshape(-1)
        neg_movie_idx_flat = neg_movie_idx.reshape(-1)
        neg_genome_flat = neg_genome.reshape(-1, neg_genome.shape[-1])
        neg_score = model(user_idx_expanded, neg_movie_idx_flat, neg_genome_flat).reshape(batch_size, neg_samples)

        loss = bpr_loss(pos_score, neg_score)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        n_examples += batch_size

    return total_loss / n_examples


def plot_training_curve(history, path):
    epochs = range(1, len(history["train_bpr_loss"]) + 1)
    colors = sns.color_palette(PALETTE, 5)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, history["train_bpr_loss"], label="train BPR loss", color=colors[3])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BPR loss")
    ax1.set_title("Recommender Training Curve (BPR, Version 3)")

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_ndcg"], label=f"val NDCG@{K}", color="firebrick", linestyle="--")
    ax2.plot(epochs, history["val_precision"], label=f"val Precision@{K}", color="darkorange", linestyle=":")
    ax2.plot(epochs, history["val_recall"], label=f"val Recall@{K}", color="seagreen", linestyle="-.")
    ax2.set_ylabel("Ranking metric")

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
    dropout=DROPOUT,
    lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    max_epochs=MAX_EPOCHS,
    patience=EARLY_STOPPING_PATIENCE,
    neg_samples=NEG_SAMPLES,
    val_neg_samples=VAL_NEG_SAMPLES,
    save_model=True,
    dataset_class=CineIQBPRDataset,
    chunk_size=None,
    rated_history_path=None,
):
    """dataset_class/chunk_size let a caller swap in CineIQBPRIterableDataset
    (streamed, chunked pyarrow reads -- for memory-constrained training
    instances) instead of the default CineIQBPRDataset (loads all positive
    interactions into memory up front) without touching the model/loss/
    optimizer/eval logic below, which is identical either way.

    rated_history_path controls what "the user has never rated" is checked
    against for negative sampling. Defaults to rl_features.parquet (the
    fullest available interaction log) if that file exists next to the
    features directory, else falls back to [train_path, val_path] -- e.g.
    inside a SageMaker container where only specific channeled files exist.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if rated_history_path is None:
        full_history_path = FEATURES_DIR / "rl_features.parquet"
        rated_history_path = full_history_path if full_history_path.exists() else [train_path, val_path]
    print(f"Negative sampling excludes ratings from: {rated_history_path}")

    streaming = issubclass(dataset_class, IterableDataset)
    extra_kwargs = {"neg_samples": neg_samples}
    if streaming:
        extra_kwargs["batch_size"] = batch_size
        if chunk_size is not None:
            extra_kwargs["chunk_size"] = chunk_size

    print(f"\nLoading BPR training dataset ({'streamed/chunked' if streaming else 'full in-memory'})...")
    train_dataset = dataset_class(
        train_path,
        user_features_path,
        movie_features_path,
        rated_history_path=rated_history_path,
        **extra_kwargs,
    )
    print(f"  {len(train_dataset):,} positive interactions x {neg_samples} negatives/positive")
    print(
        f"  users: {train_dataset.num_users:,}  movies: {train_dataset.num_movies:,}  "
        f"genome_dim: {train_dataset.genome_dim}"
    )

    if streaming:
        train_loader = DataLoader(train_dataset, batch_size=None)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = NCF(
        num_users=train_dataset.num_users,
        num_movies=train_dataset.num_movies,
        embedding_dim=embedding_dim,
        genome_dim=train_dataset.genome_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Validation ranking-eval setup: negatives sampled once (fixed seed) and
    # reused every epoch, so metric trajectories reflect model improvement
    # rather than sampling variance; only the scoring (forward pass) reruns
    # each epoch against the current weights.
    # ------------------------------------------------------------------
    print("\nBuilding validation ranking-eval candidates (negatives fixed across epochs)...")
    val_positives = pd.read_parquet(val_path, columns=["userId", "movieId"])
    val_positives["is_positive"] = True
    val_user_ids = val_positives["userId"].unique()

    val_history = load_rating_history(rated_history_path, val_user_ids)
    movie_features_full = pd.read_parquet(movie_features_path, columns=["movieId"])
    catalog = movie_features_full["movieId"].to_numpy()
    val_negatives = sample_negatives(val_user_ids, val_history, catalog, n_neg=val_neg_samples)
    val_negatives["is_positive"] = False

    val_candidates_base = pd.concat([val_positives, val_negatives], ignore_index=True)
    print(f"  {len(val_user_ids):,} val users, {len(val_candidates_base):,} total val candidates")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = {"train_bpr_loss": [], "val_ndcg": [], "val_precision": [], "val_recall": []}
    best_val_ndcg = -float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)

        val_scored = score_candidates(
            model, val_candidates_base, train_dataset.user_id_map, train_dataset.movie_id_map,
            train_dataset.genome_lookup, device,
        )
        val_metrics = per_user_ranking_metrics_negsample(val_scored, k=K)
        val_ndcg = float(val_metrics["ndcg_10"].mean())
        val_precision = float(val_metrics["precision_10"].mean())
        val_recall = float(val_metrics["recall_10"].mean(skipna=True))

        history["train_bpr_loss"].append(train_loss)
        history["val_ndcg"].append(val_ndcg)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        print(
            f"Epoch {epoch:3d} | BPR loss: {train_loss:.4f} | val NDCG@10: {val_ndcg:.4f} | "
            f"val Precision@10: {val_precision:.4f} | val Recall@10: {val_recall:.4f}"
        )

        if val_ndcg > best_val_ndcg + 1e-4:
            best_val_ndcg = val_ndcg
            epochs_without_improvement = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "num_users": train_dataset.num_users,
                "num_movies": train_dataset.num_movies,
                "embedding_dim": embedding_dim,
                "genome_dim": train_dataset.genome_dim,
                "hidden_layers": list(hidden_layers),
                "dropout": dropout,
                "user_id_map": train_dataset.user_id_map,
                "movie_id_map": train_dataset.movie_id_map,
                "epoch": epoch,
                "train_bpr_loss": train_loss,
                "val_ndcg": val_ndcg,
                "val_precision": val_precision,
                "val_recall": val_recall,
            }
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no val NDCG@10 improvement for {patience} epochs)")
                break

    if save_model:
        model_output_path = Path(model_output_path)
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, model_output_path)
        print(
            f"Saved best model (epoch {best_state['epoch']}, val NDCG@10 {best_state['val_ndcg']:.4f}, "
            f"val Precision@10 {best_state['val_precision']:.4f}) -> {model_output_path}"
        )
    else:
        print(
            f"save_model=False -- skipping checkpoint save "
            f"(best epoch {best_state['epoch']}, val NDCG@10 {best_state['val_ndcg']:.4f})"
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
