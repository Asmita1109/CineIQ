"""Evaluate the trained NCF recommender on the held-out test set, using
negative sampling for a proper (non-degenerate) top-10-out-of-N ranking
evaluation.

The previous version of this script ranked only each user's 5 known
leave-last-5-out test items against each other -- with just 5 candidates and
k=10, "top-10" degenerated to "all 5, ranked," which makes Recall@10
trivially 1.0 always (there's nothing to exclude) and is not a meaningful
Precision/Recall signal. This version follows the standard implicit-feedback
protocol from the original NCF paper (He et al. 2017): for each test user,
sample 100 movies they've never rated as negatives, mix them with their 5
real held-out movies (105 candidates total), rank all 105 by predicted
score, and see whether the real movies surface in the top 10. Relevance for
NDCG/Precision/Recall is binary here (real test movie vs. sampled negative)
-- RMSE/MAE are unaffected and still computed on the 5 real ratings only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import NCF

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "recommender_model.pt"
TRAINING_RESULTS_PATH = MODELS_DIR / "training_results.json"
OUTPUT_PATH = MODELS_DIR / "recommender_test_results.json"

SCORE_BATCH_SIZE = 4096
K = 10
N_NEGATIVES = 100
OVERSAMPLE_FACTOR = 1.6  # sample this many extra candidates per user before filtering out true positives
RELEVANCE_THRESHOLD = 4.0  # for the old RMSE-based Precision/Recall, unused by the new binary-relevance version
TOP_N_GENRES = 5
RANDOM_SEED = 42


def load_checkpoint():
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = NCF(
        num_users=ckpt["num_users"],
        num_movies=ckpt["num_movies"],
        embedding_dim=ckpt["embedding_dim"],
        genome_dim=ckpt["genome_dim"],
        hidden_layers=ckpt["hidden_layers"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def build_genome_lookup(movie_features, movie_id_map, genome_dim):
    """Dense array indexed by mapped movie index, so scoring can gather
    genome vectors for arbitrary candidate movies with a fast numpy
    fancy-index instead of a per-batch merge."""
    genome_cols = [f"genome_emb_{i}" for i in range(genome_dim)]
    ids_in_order = [None] * len(movie_id_map)
    for movie_id, idx in movie_id_map.items():
        ids_in_order[idx] = movie_id
    aligned = movie_features.set_index("movieId")[genome_cols].reindex(ids_in_order).fillna(0.0)
    return aligned.to_numpy(dtype="float32")


def build_user_history(test_user_ids):
    """Every movie each test user has ever actually rated (across
    train/val/test), so negative sampling never accidentally samples a true
    positive."""
    rl = pd.read_parquet(FEATURES_DIR / "rl_features.parquet", columns=["userId", "movieId"])
    return rl[rl["userId"].isin(set(test_user_ids))]


def sample_negatives(test_user_ids, user_history, catalog, n_neg=N_NEGATIVES, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n_users = len(test_user_ids)
    oversample = min(len(catalog), int(n_neg * OVERSAMPLE_FACTOR) + 10)

    sampled_idx = np.array(
        [rng.choice(len(catalog), size=oversample, replace=False) for _ in range(n_users)]
    )
    sampled_movies = catalog[sampled_idx]

    candidates = pd.DataFrame(
        {"userId": np.repeat(test_user_ids, oversample), "movieId": sampled_movies.ravel()}
    )
    candidates = candidates.merge(
        user_history.assign(_is_true_positive=True), on=["userId", "movieId"], how="left"
    )
    candidates = candidates[candidates["_is_true_positive"].isna()].drop(columns=["_is_true_positive"])

    candidates["_rank"] = candidates.groupby("userId").cumcount()
    negatives = candidates[candidates["_rank"] < n_neg].drop(columns=["_rank"])

    counts = negatives.groupby("userId").size()
    short = counts[counts < n_neg]
    if len(short):
        print(
            f"  Warning: {len(short)} users had fewer than {n_neg} valid negatives after "
            f"{oversample}x oversampling -- using however many were available for them."
        )
    return negatives


@torch.no_grad()
def score_candidates(model, candidates, user_id_map, movie_id_map, genome_lookup, device, batch_size=SCORE_BATCH_SIZE):
    user_idx = candidates["userId"].map(user_id_map).to_numpy(dtype="int64")
    movie_idx = candidates["movieId"].map(movie_id_map).to_numpy(dtype="int64")
    genome = genome_lookup[movie_idx]

    n = len(candidates)
    preds = np.empty(n, dtype="float32")
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        u = torch.from_numpy(user_idx[start:end]).to(device)
        m = torch.from_numpy(movie_idx[start:end]).to(device)
        g = torch.from_numpy(genome[start:end]).to(device)
        preds[start:end] = model(u, m, g).cpu().numpy()

    candidates = candidates.copy()
    candidates["pred_score"] = preds
    return candidates


def rmse_mae(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return rmse, mae


def ndcg_at_k_vectorized(relevance_matrix, pred_matrix, k):
    n_users, n_items = relevance_matrix.shape
    k_eff = min(k, n_items)

    pred_rank_idx = np.argsort(-pred_matrix, axis=1)[:, :k_eff]
    gains = np.take_along_axis(relevance_matrix, pred_rank_idx, axis=1)
    discounts = 1.0 / np.log2(np.arange(2, k_eff + 2))
    dcg = (gains * discounts).sum(axis=1)

    ideal_rank_idx = np.argsort(-relevance_matrix, axis=1)[:, :k_eff]
    ideal_gains = np.take_along_axis(relevance_matrix, ideal_rank_idx, axis=1)
    idcg = (ideal_gains * discounts).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(idcg > 0, dcg / idcg, 0.0)


def per_user_ranking_metrics_old(rec_test_scored, k=K, relevance_threshold=RELEVANCE_THRESHOLD):
    """Original methodology: rank only the 5 known test items against each
    other. Recall@10 is trivially 1.0 under this design (kept only for the
    old-vs-new comparison table)."""
    df = rec_test_scored.sort_values(["userId", "movieId"])
    user_ids = df["userId"].to_numpy()
    unique_users, counts = np.unique(user_ids, return_counts=True)
    group_size = counts[0]
    if not np.all(counts == group_size):
        raise ValueError("per_user_ranking_metrics_old expects a fixed number of items per user")

    n_users = len(unique_users)
    true_matrix = df["rating"].to_numpy().reshape(n_users, group_size)
    pred_matrix = df["pred_score"].to_numpy().reshape(n_users, group_size)
    k_eff = min(k, group_size)

    ndcg = ndcg_at_k_vectorized(true_matrix, pred_matrix, k)

    rank_order = np.argsort(-pred_matrix, axis=1)[:, :k_eff]
    relevant = true_matrix >= relevance_threshold
    relevant_in_topk = np.take_along_axis(relevant, rank_order, axis=1)
    n_relevant_topk = relevant_in_topk.sum(axis=1)
    n_relevant_total = relevant.sum(axis=1)

    precision = n_relevant_topk / k_eff
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(n_relevant_total > 0, n_relevant_topk / np.maximum(n_relevant_total, 1), np.nan)

    return pd.DataFrame(
        {"userId": unique_users, "ndcg_10": ndcg, "precision_10": precision, "recall_10": recall}
    )


def per_user_ranking_metrics_negsample(candidates, k=K):
    """New methodology: rank each user's 5 real + up to 100 sampled negative
    movies (binary relevance) and see whether the real ones surface in the
    top 10 -- a proper top-N-out-of-many evaluation."""
    counts = candidates.groupby("userId").size()
    full_size = int(counts.mode().iloc[0])
    valid_users = counts[counts == full_size].index
    dropped = len(counts) - len(valid_users)
    if dropped:
        print(
            f"  Note: {dropped} users had a candidate count other than the typical {full_size} "
            f"(rare negative-sampling shortfall) -- excluded from ranking metrics."
        )

    df = candidates[candidates["userId"].isin(valid_users)].sort_values(["userId", "movieId"])
    user_ids = df["userId"].to_numpy()
    unique_users = np.unique(user_ids)
    n_users = len(unique_users)

    relevance_matrix = df["is_positive"].to_numpy().astype("float64").reshape(n_users, full_size)
    pred_matrix = df["pred_score"].to_numpy().reshape(n_users, full_size)
    k_eff = min(k, full_size)

    ndcg = ndcg_at_k_vectorized(relevance_matrix, pred_matrix, k)

    rank_order = np.argsort(-pred_matrix, axis=1)[:, :k_eff]
    relevant_bool = relevance_matrix.astype(bool)
    relevant_in_topk = np.take_along_axis(relevant_bool, rank_order, axis=1)
    n_relevant_topk = relevant_in_topk.sum(axis=1)
    n_relevant_total = relevant_bool.sum(axis=1)

    precision = n_relevant_topk / k_eff
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(n_relevant_total > 0, n_relevant_topk / np.maximum(n_relevant_total, 1), np.nan)

    return pd.DataFrame(
        {"userId": unique_users, "ndcg_10": ndcg, "precision_10": precision, "recall_10": recall}
    )


def load_movie_genre_pairs(movie_features):
    genre_pairs = movie_features[["movieId", "genres"]].copy()
    genre_pairs = genre_pairs[genre_pairs["genres"] != "(no genres listed)"]
    genre_pairs = genre_pairs.assign(genre=genre_pairs["genres"].str.split("|")).explode("genre")
    return genre_pairs[["movieId", "genre"]].reset_index(drop=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint: {MODEL_PATH}")
    model, ckpt = load_checkpoint()
    model.to(device)
    print(f"  Checkpoint epoch {ckpt['epoch']}, val RMSE {ckpt['val_rmse']:.4f}, val NDCG@10 {ckpt['val_ndcg']:.4f}")

    print("\nLoading rec_test.parquet, user_features.parquet, movie_features.parquet...")
    rec_test = pd.read_parquet(FEATURES_DIR / "rec_test.parquet", columns=["userId", "movieId", "rating"])
    user_features = pd.read_parquet(FEATURES_DIR / "user_features.parquet", columns=["userId", "user_segment"])
    movie_features = pd.read_parquet(FEATURES_DIR / "movie_features.parquet")
    test_user_ids = rec_test["userId"].unique()
    print(f"  rec_test: {rec_test.shape}, {len(test_user_ids):,} users")

    genome_lookup = build_genome_lookup(movie_features, ckpt["movie_id_map"], ckpt["genome_dim"])

    # ------------------------------------------------------------------
    # 1-2. Negative sampling: 100 never-rated movies per test user
    # ------------------------------------------------------------------
    print(f"\nBuilding full rating history for {len(test_user_ids):,} test users (to exclude from sampling)...")
    user_history = build_user_history(test_user_ids)
    print(f"  {len(user_history):,} historical (user, movie) pairs")

    print(f"Sampling {N_NEGATIVES} negatives per user...")
    catalog = movie_features["movieId"].to_numpy()
    negatives = sample_negatives(test_user_ids, user_history, catalog)
    print(f"  {len(negatives):,} negative candidates sampled ({len(negatives) / len(test_user_ids):.1f}/user avg)")

    # ------------------------------------------------------------------
    # 3. Combine 5 real + up to 100 negative candidates, score all of them
    # ------------------------------------------------------------------
    positives = rec_test.copy()
    positives["is_positive"] = True
    negatives = negatives.copy()
    negatives["rating"] = np.float32(np.nan)
    negatives["is_positive"] = False
    candidates = pd.concat([positives, negatives], ignore_index=True)
    print(f"\nScoring {len(candidates):,} total candidates ({len(positives):,} real + {len(negatives):,} negative)...")
    candidates = score_candidates(model, candidates, ckpt["user_id_map"], ckpt["movie_id_map"], genome_lookup, device)

    # ------------------------------------------------------------------
    # 4. Metrics: RMSE/MAE unchanged (real ratings only); NDCG/Precision/Recall
    #    now computed properly (5 real ranked among 105)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION (negative-sampled: 5 real + 100 negatives)")
    print("=" * 70)
    real_scored = candidates[candidates["is_positive"]]
    test_rmse, test_mae = rmse_mae(real_scored["rating"].to_numpy(), real_scored["pred_score"].to_numpy())

    old_per_user = per_user_ranking_metrics_old(real_scored)
    new_per_user = per_user_ranking_metrics_negsample(candidates)

    old_ndcg, old_precision, old_recall = (
        float(old_per_user["ndcg_10"].mean()),
        float(old_per_user["precision_10"].mean()),
        float(old_per_user["recall_10"].mean(skipna=True)),
    )
    new_ndcg, new_precision, new_recall = (
        float(new_per_user["ndcg_10"].mean()),
        float(new_per_user["precision_10"].mean()),
        float(new_per_user["recall_10"].mean(skipna=True)),
    )

    print(f"RMSE:         {test_rmse:.4f}  (unchanged either way)")
    print(f"MAE:          {test_mae:.4f}  (unchanged either way)")
    print(f"NDCG@10:      {new_ndcg:.4f}  (was {old_ndcg:.4f} under the old 5-item-only ranking)")
    print(f"Precision@10: {new_precision:.4f}  (was {old_precision:.4f})")
    print(f"Recall@10:    {new_recall:.4f}  (was {old_recall:.4f} -- trivially 1.0 under the old design)")

    # ------------------------------------------------------------------
    # 5. RMSE by user segment (unaffected by negative sampling)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RMSE BY USER SEGMENT")
    print("=" * 70)
    segment_preds = real_scored.merge(user_features, on="userId", how="left")
    segment_rmse = {}
    for segment, group in segment_preds.groupby("user_segment", observed=True):
        seg_rmse, seg_mae = rmse_mae(group["rating"].to_numpy(), group["pred_score"].to_numpy())
        segment_rmse[str(segment)] = {"rmse": seg_rmse, "mae": seg_mae, "n_interactions": int(len(group))}
        print(f"  {segment:<10} RMSE = {seg_rmse:.4f}   MAE = {seg_mae:.4f}   (n={len(group):,})")

    # ------------------------------------------------------------------
    # NDCG@10 by top genres (using the new negative-sampled per-user NDCG)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"NDCG@10 BY TOP {TOP_N_GENRES} GENRES (negative-sampled)")
    print("=" * 70)
    genre_pairs = load_movie_genre_pairs(movie_features)
    test_movie_genres = rec_test[["movieId"]].drop_duplicates().merge(genre_pairs, on="movieId", how="inner")
    top_genres = test_movie_genres["genre"].value_counts().head(TOP_N_GENRES).index.tolist()
    print(f"Top {TOP_N_GENRES} genres in the test set: {top_genres}")
    print("(each user's negative-sampled NDCG@10 is averaged across every genre found among their real test movies)")

    user_genre_pairs = rec_test[["userId", "movieId"]].merge(genre_pairs, on="movieId", how="inner")
    new_ndcg_lookup = new_per_user.set_index("userId")["ndcg_10"]

    genre_ndcg = {}
    for genre in top_genres:
        genre_user_ids = user_genre_pairs.loc[user_genre_pairs["genre"] == genre, "userId"].unique()
        scores = new_ndcg_lookup.reindex(genre_user_ids).dropna()
        genre_ndcg[genre] = {"ndcg_10": float(scores.mean()), "n_users": int(len(scores))}
        print(f"  {genre:<15} NDCG@10 = {scores.mean():.4f}   (n_users={len(scores):,})")

    # ------------------------------------------------------------------
    # 6. Old vs. new comparison table, plus val (from training) vs test
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OLD (5-item-only) vs NEW (negative-sampled 105) -- COMPARISON")
    print("=" * 70)
    comparison = pd.DataFrame(
        {
            "Metric": ["RMSE", "MAE", "NDCG@10", "Precision@10", "Recall@10"],
            "Old (5 items)": [test_rmse, test_mae, old_ndcg, old_precision, old_recall],
            "New (105, negatives)": [test_rmse, test_mae, new_ndcg, new_precision, new_recall],
        }
    )
    print(comparison.to_string(index=False))

    val_rmse = val_ndcg = None
    if TRAINING_RESULTS_PATH.exists():
        with open(TRAINING_RESULTS_PATH) as f:
            training_results = json.load(f)
        val_rmse = training_results["best_epoch"]["val_rmse"]
        val_ndcg = training_results["best_epoch"]["val_ndcg_10"]

    print("\n" + "=" * 70)
    print("VAL (TRAINING) vs TEST (NEW METHODOLOGY) -- FINAL SUMMARY")
    print("=" * 70)
    summary = pd.DataFrame(
        {
            "Metric": ["RMSE", "MAE", "NDCG@10", "Precision@10", "Recall@10"],
            "Val (best epoch)": [val_rmse, None, val_ndcg, None, None],
            "Test": [test_rmse, test_mae, new_ndcg, new_precision, new_recall],
        }
    )
    print(summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    results = {
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_rmse": ckpt["val_rmse"],
        "checkpoint_val_ndcg_10": ckpt["val_ndcg"],
        "methodology": {
            "n_negatives_per_user": N_NEGATIVES,
            "k": K,
            "relevance": "binary (real held-out movie vs. sampled negative)",
        },
        "test_metrics": {
            "rmse": test_rmse,
            "mae": test_mae,
            "ndcg_10": new_ndcg,
            "precision_10": new_precision,
            "recall_10": new_recall,
            "n_users": int(len(test_user_ids)),
        },
        "old_vs_new_ranking_metrics": {
            "old_5_item_only": {"ndcg_10": old_ndcg, "precision_10": old_precision, "recall_10": old_recall},
            "new_negative_sampled": {"ndcg_10": new_ndcg, "precision_10": new_precision, "recall_10": new_recall},
        },
        "rmse_by_user_segment": segment_rmse,
        "ndcg_10_by_top_genres": genre_ndcg,
        "val_vs_test": {
            "val_rmse": val_rmse,
            "val_ndcg_10": val_ndcg,
            "test_rmse": test_rmse,
            "test_ndcg_10": new_ndcg,
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
