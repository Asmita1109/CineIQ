"""Evaluate the trained NCF recommender on the held-out test set -- Version 3
(BPR raw relevance scores, ranking metrics only).

The model no longer predicts a calibrated 0.5-5 star rating (model.py's
Version 3 NCF dropped the sigmoid/rating-scale output in favor of BPR
pairwise ranking), so RMSE/MAE against real ratings would be meaningless and
have been removed. This follows the standard implicit-feedback protocol from
the original NCF paper (He et al. 2017): for each test user, sample 100
movies they've never rated as negatives, mix them with their 5 real
held-out movies (105 candidates total), rank all 105 by the model's raw
relevance score, and see whether the real movies surface in the top 10.
Relevance for NDCG/Precision/Recall is binary (real test movie vs. sampled
negative).

sample_negatives/score_candidates/per_user_ranking_metrics_negsample are
also imported by train.py, which reuses this exact methodology for its
per-epoch validation metrics -- keep their signatures stable.
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
OUTPUT_PATH = MODELS_DIR / "recommender_test_results.json"

SCORE_BATCH_SIZE = 4096
K = 10
N_NEGATIVES = 100
OVERSAMPLE_FACTOR = 1.6  # sample this many extra candidates per user before filtering out true positives
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
        dropout=ckpt.get("dropout", 0.2),
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
    """Scores every (userId, movieId) candidate with the model's raw
    relevance score -- not a rating prediction. Higher just means "more
    relevant"; the value itself isn't on any fixed scale."""
    user_idx = candidates["userId"].map(user_id_map).to_numpy(dtype="int64")
    movie_idx = candidates["movieId"].map(movie_id_map).to_numpy(dtype="int64")
    genome = genome_lookup[movie_idx]

    n = len(candidates)
    scores = np.empty(n, dtype="float32")
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        u = torch.from_numpy(user_idx[start:end]).to(device)
        m = torch.from_numpy(movie_idx[start:end]).to(device)
        g = torch.from_numpy(genome[start:end]).to(device)
        scores[start:end] = model(u, m, g).cpu().numpy()

    candidates = candidates.copy()
    candidates["pred_score"] = scores
    return candidates


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


def per_user_ranking_metrics_negsample(candidates, k=K):
    """Ranks each user's 5 real + up to 100 sampled negative movies (binary
    relevance) and checks whether the real ones surface in the top 10 -- a
    proper top-N-out-of-many evaluation, not just reordering 5 known items."""
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


def summarize_by_group(per_user_metrics, group_lookup, group_col):
    """Merge per-user ranking metrics against a (userId -> group value)
    lookup and average NDCG@10/Precision@10/Recall@10 within each group."""
    merged = per_user_metrics.merge(group_lookup, on="userId", how="left")
    summary = {}
    for group_value, rows in merged.groupby(group_col, observed=True):
        summary[str(group_value)] = {
            "ndcg_10": float(rows["ndcg_10"].mean()),
            "precision_10": float(rows["precision_10"].mean()),
            "recall_10": float(rows["recall_10"].mean(skipna=True)),
            "n_users": int(len(rows)),
        }
    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint: {MODEL_PATH}")
    model, ckpt = load_checkpoint()
    model.to(device)
    val_ndcg_ckpt = ckpt.get("val_ndcg")
    val_precision_ckpt = ckpt.get("val_precision")
    val_recall_ckpt = ckpt.get("val_recall")
    ckpt_summary = f"  Checkpoint epoch {ckpt['epoch']}"
    if val_ndcg_ckpt is not None:
        ckpt_summary += f", val NDCG@10 {val_ndcg_ckpt:.4f}"
    if val_precision_ckpt is not None:
        ckpt_summary += f", val Precision@10 {val_precision_ckpt:.4f}"
    if val_recall_ckpt is not None:
        ckpt_summary += f", val Recall@10 {val_recall_ckpt:.4f}"
    print(ckpt_summary)

    print("\nLoading rec_test.parquet, user_features.parquet, movie_features.parquet...")
    rec_test = pd.read_parquet(FEATURES_DIR / "rec_test.parquet", columns=["userId", "movieId"])
    user_features = pd.read_parquet(FEATURES_DIR / "user_features.parquet", columns=["userId", "user_segment"])
    movie_features = pd.read_parquet(FEATURES_DIR / "movie_features.parquet")
    test_user_ids = rec_test["userId"].unique()
    print(f"  rec_test: {rec_test.shape}, {len(test_user_ids):,} users")

    genome_lookup = build_genome_lookup(movie_features, ckpt["movie_id_map"], ckpt["genome_dim"])

    # ------------------------------------------------------------------
    # 3. Negative sampling: 100 never-rated movies per test user
    # ------------------------------------------------------------------
    print(f"\nBuilding full rating history for {len(test_user_ids):,} test users (to exclude from sampling)...")
    user_history = build_user_history(test_user_ids)
    print(f"  {len(user_history):,} historical (user, movie) pairs")

    print(f"Sampling {N_NEGATIVES} negatives per user...")
    catalog = movie_features["movieId"].to_numpy()
    negatives = sample_negatives(test_user_ids, user_history, catalog)
    print(f"  {len(negatives):,} negative candidates sampled ({len(negatives) / len(test_user_ids):.1f}/user avg)")

    # ------------------------------------------------------------------
    # 4. Combine 5 real + up to 100 negative candidates; score with the
    #    model's raw relevance score (not a rating prediction)
    # ------------------------------------------------------------------
    positives = rec_test.copy()
    positives["is_positive"] = True
    negatives = negatives.copy()
    negatives["is_positive"] = False
    candidates = pd.concat([positives, negatives], ignore_index=True)
    print(f"\nScoring {len(candidates):,} total candidates ({len(positives):,} real + {len(negatives):,} negative)...")
    candidates = score_candidates(model, candidates, ckpt["user_id_map"], ckpt["movie_id_map"], genome_lookup, device)

    # ------------------------------------------------------------------
    # 1-2. Ranking metrics only: NDCG@10, Precision@10, Recall@10
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION (negative-sampled: 5 real + 100 negatives)")
    print("=" * 70)
    per_user = per_user_ranking_metrics_negsample(candidates)
    test_ndcg = float(per_user["ndcg_10"].mean())
    test_precision = float(per_user["precision_10"].mean())
    test_recall = float(per_user["recall_10"].mean(skipna=True))

    print(f"NDCG@10:      {test_ndcg:.4f}")
    print(f"Precision@10: {test_precision:.4f}")
    print(f"Recall@10:    {test_recall:.4f}")

    # ------------------------------------------------------------------
    # 5. Ranking metrics by user segment
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RANKING METRICS BY USER SEGMENT")
    print("=" * 70)
    segment_metrics = summarize_by_group(per_user, user_features, "user_segment")
    for segment, m in segment_metrics.items():
        print(
            f"  {segment:<10} NDCG@10 = {m['ndcg_10']:.4f}   Precision@10 = {m['precision_10']:.4f}   "
            f"Recall@10 = {m['recall_10']:.4f}   (n_users={m['n_users']:,})"
        )

    # ------------------------------------------------------------------
    # NDCG@10 by top genres
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"NDCG@10 BY TOP {TOP_N_GENRES} GENRES")
    print("=" * 70)
    genre_pairs = load_movie_genre_pairs(movie_features)
    test_movie_genres = rec_test[["movieId"]].drop_duplicates().merge(genre_pairs, on="movieId", how="inner")
    top_genres = test_movie_genres["genre"].value_counts().head(TOP_N_GENRES).index.tolist()
    print(f"Top {TOP_N_GENRES} genres in the test set: {top_genres}")
    print("(each user's NDCG@10 is averaged across every genre found among their real test movies)")

    user_genre_pairs = rec_test[["userId", "movieId"]].merge(genre_pairs, on="movieId", how="inner")
    ndcg_lookup = per_user.set_index("userId")["ndcg_10"]

    genre_ndcg = {}
    for genre in top_genres:
        genre_user_ids = user_genre_pairs.loc[user_genre_pairs["genre"] == genre, "userId"].unique()
        scores = ndcg_lookup.reindex(genre_user_ids).dropna()
        genre_ndcg[genre] = {"ndcg_10": float(scores.mean()), "n_users": int(len(scores))}
        print(f"  {genre:<15} NDCG@10 = {scores.mean():.4f}   (n_users={len(scores):,})")

    # ------------------------------------------------------------------
    # 6. Results table -- ranking metrics only, val (from checkpoint) vs test
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VAL (FROM CHECKPOINT) vs TEST -- FINAL SUMMARY")
    print("=" * 70)
    summary = pd.DataFrame(
        {
            "Metric": ["NDCG@10", "Precision@10", "Recall@10"],
            "Val (best epoch)": [val_ndcg_ckpt, val_precision_ckpt, val_recall_ckpt],
            "Test": [test_ndcg, test_precision, test_recall],
        }
    )
    print(summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    results = {
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_ndcg_10": val_ndcg_ckpt,
        "checkpoint_val_precision_10": val_precision_ckpt,
        "checkpoint_val_recall_10": val_recall_ckpt,
        "methodology": {
            "n_negatives_per_user": N_NEGATIVES,
            "k": K,
            "relevance": "binary (real held-out movie vs. sampled negative)",
            "scoring": "raw NCF relevance score (Version 3 BPR model, no rating scale)",
        },
        "test_metrics": {
            "ndcg_10": test_ndcg,
            "precision_10": test_precision,
            "recall_10": test_recall,
            "n_users": int(len(test_user_ids)),
        },
        "ranking_metrics_by_user_segment": segment_metrics,
        "ndcg_10_by_top_genres": genre_ndcg,
        "val_vs_test": {
            "val_ndcg_10": val_ndcg_ckpt,
            "val_precision_10": val_precision_ckpt,
            "val_recall_10": val_recall_ckpt,
            "test_ndcg_10": test_ndcg,
            "test_precision_10": test_precision,
            "test_recall_10": test_recall,
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
