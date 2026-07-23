"""
CineIQ -- Feature Engineering

Builds three feature tables from 33.8M real movie ratings (1995-2023):
- user_features.parquet: one row per user, captures rating behavior and preferences
- movie_features.parquet: one row per movie, includes genre and content embeddings
- rl_features.parquet: full interaction log used to train the RL agent

Input: data/processed/ | Output: data/features/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import TruncatedSVD

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

CHUNK_SIZE_THRESHOLD_BYTES = 1_000_000_000  # 1 GB -- only chunk files larger than this
CHUNK_SIZE = 500_000
RATING_POSITIVE_THRESHOLD = 4.0
CASUAL_MAX = 20
REGULAR_MAX = 100
GENOME_SVD_COMPONENTS = 50
RECENT_HISTORY_LEN = 5

RATINGS_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
    "year": "int16",
    "month": "int8",
}
GENOME_DTYPES = {"movieId": "int32", "tagId": "int16", "relevance": "float32"}


def ratings_chunks():
    path = PROCESSED_DIR / "ratings_clean.csv"
    size_gb = path.stat().st_size / 1e9
    if path.stat().st_size > CHUNK_SIZE_THRESHOLD_BYTES:
        print(f"ratings_clean.csv is {size_gb:.2f} GB (> 1GB) -- streaming in {CHUNK_SIZE:,}-row chunks")
        yield from pd.read_csv(path, dtype=RATINGS_DTYPES, chunksize=CHUNK_SIZE)
    else:
        print(f"ratings_clean.csv is {size_gb:.2f} GB (<= 1GB) -- loading in a single pass")
        yield pd.read_csv(path, dtype=RATINGS_DTYPES)


def load_movie_genres():
    movies = pd.read_csv(PROCESSED_DIR / "movies_clean.csv")
    genre_pairs = movies[["movieId", "genres"]].copy()
    genre_pairs = genre_pairs[genre_pairs["genres"] != "(no genres listed)"]
    genre_pairs = genre_pairs.assign(genre=genre_pairs["genres"].str.split("|")).explode("genre")
    return movies, genre_pairs[["movieId", "genre"]].reset_index(drop=True)


def segment_label(n):
    if n <= CASUAL_MAX:
        return "casual"
    elif n <= REGULAR_MAX:
        return "regular"
    return "power"


def show(name, df):
    print(f"\n{name}: shape = {df.shape}")
    print(df.head())


def report_output_file(path, df=None):
    file_size_bytes = path.stat().st_size
    print(f"\n{path.name}")
    print(f"  File size on disk: {file_size_bytes / 1e6:,.1f} MB")
    if df is None:
        df = pd.read_parquet(path)
    mem = df.memory_usage(deep=True).sum()
    print(f"  Shape: {df.shape}")
    print(f"  In-memory usage: {mem / 1e6:,.1f} MB")
    print("  Sample rows:")
    print(df.head(5).to_string(index=False))


# ------------------------------------------------------------------
# Genome embeddings: stream genome_scores_clean.csv in chunks to build the
# movie x tag matrix, then compress 1,128 dims -> 50 via SVD.
# ------------------------------------------------------------------
def build_genome_embeddings(n_components=GENOME_SVD_COMPONENTS):
    path = PROCESSED_DIR / "genome_scores_clean.csv"

    print(f"Pass A - discovering movieId/tagId sets from {path.name} in {CHUNK_SIZE:,}-row chunks")
    movie_ids = set()
    tag_ids = set()
    for chunk in pd.read_csv(path, dtype=GENOME_DTYPES, usecols=["movieId", "tagId"], chunksize=CHUNK_SIZE):
        movie_ids.update(chunk["movieId"].unique().tolist())
        tag_ids.update(chunk["tagId"].unique().tolist())
    movie_ids = sorted(movie_ids)
    tag_ids = sorted(tag_ids)
    movie_index = {m: i for i, m in enumerate(movie_ids)}
    tag_index = {t: i for i, t in enumerate(tag_ids)}
    print(f"  {len(movie_ids):,} movies x {len(tag_ids):,} tags")

    matrix = np.zeros((len(movie_ids), len(tag_ids)), dtype="float32")

    print(f"Pass B - filling the movie x tag matrix in {CHUNK_SIZE:,}-row chunks")
    for chunk in pd.read_csv(path, dtype=GENOME_DTYPES, chunksize=CHUNK_SIZE):
        rows = chunk["movieId"].map(movie_index).to_numpy()
        cols = chunk["tagId"].map(tag_index).to_numpy()
        matrix[rows, cols] = chunk["relevance"].to_numpy(dtype="float32")

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embedding = svd.fit_transform(matrix)
    explained = svd.explained_variance_ratio_.sum()
    print(f"  SVD: {len(tag_ids)} dims -> {n_components} dims (explained variance ratio: {explained:.3f})")

    cols_out = [f"genome_emb_{i}" for i in range(n_components)]
    embedding_df = pd.DataFrame(embedding, columns=cols_out).astype("float32").round(6)
    embedding_df.insert(0, "movieId", movie_ids)
    return embedding_df, cols_out


# ------------------------------------------------------------------
# Pass 1: stream ratings_clean.csv once, accumulate every aggregate
# ------------------------------------------------------------------
def run_pass_one(genre_pairs):
    user_partials = []
    movie_partials = []
    user_genre_partials = []
    recent_hist = {}

    n_chunks = 0
    n_rows = 0
    for chunk in ratings_chunks():
        n_chunks += 1
        n_rows += len(chunk)
        chunk = chunk.copy()
        chunk["rating_sq"] = chunk["rating"].astype("float64") ** 2
        chunk["reward"] = (chunk["rating"] >= RATING_POSITIVE_THRESHOLD).astype("int32")

        user_partials.append(
            chunk.groupby("userId").agg(
                count=("rating", "size"),
                sum=("rating", "sum"),
                sumsq=("rating_sq", "sum"),
                min=("rating", "min"),
                max=("rating", "max"),
                reward_sum=("reward", "sum"),
            )
        )
        movie_partials.append(
            chunk.groupby("movieId").agg(
                count=("rating", "size"), sum=("rating", "sum"), sumsq=("rating_sq", "sum")
            )
        )

        chunk_genre = chunk.merge(genre_pairs, on="movieId", how="inner")
        user_genre_partials.append(
            chunk_genre.groupby(["userId", "genre"]).size().rename("count").reset_index()
        )

        top5_chunk = (
            chunk.sort_values("timestamp", ascending=False)
            .groupby("userId")
            .head(RECENT_HISTORY_LEN)[["userId", "timestamp", "movieId"]]
        )
        for uid, ts, mid in zip(
            top5_chunk["userId"].to_numpy(),
            top5_chunk["timestamp"].to_numpy(),
            top5_chunk["movieId"].to_numpy(),
        ):
            lst = recent_hist.setdefault(int(uid), [])
            lst.append((int(ts), int(mid)))
            if len(lst) > RECENT_HISTORY_LEN:
                lst.sort(key=lambda x: x[0], reverse=True)
                del lst[RECENT_HISTORY_LEN:]

        print(f"  Pass 1 - chunk {n_chunks}: {n_rows:,} rows processed so far")

    user_stats = (
        pd.concat(user_partials)
        .groupby("userId")
        .agg(count=("count", "sum"), sum=("sum", "sum"), sumsq=("sumsq", "sum"),
             min=("min", "min"), max=("max", "max"), reward_sum=("reward_sum", "sum"))
        .reset_index()
    )
    movie_stats = pd.concat(movie_partials).groupby("movieId").sum().reset_index()
    user_genre_counts = (
        pd.concat(user_genre_partials).groupby(["userId", "genre"])["count"].sum().reset_index()
    )

    for lst in recent_hist.values():
        lst.sort(key=lambda x: x[0], reverse=True)

    return user_stats, movie_stats, user_genre_counts, recent_hist


def finalize_movie_stats(raw):
    df = raw.copy()
    count = df["count"]
    mean = df["sum"] / count
    denom = (count - 1).clip(lower=1)
    var = (df["sumsq"] - count * mean**2) / denom
    var = var.clip(lower=0).where(count > 1, 0.0)

    return pd.DataFrame(
        {
            "movieId": df["movieId"],
            "total_ratings": count.astype("int32"),
            "avg_rating": mean.astype("float32"),
            "rating_std": np.sqrt(var).astype("float32"),
        }
    )


def finalize_user_stats(raw):
    df = raw.copy()
    count = df["count"]
    mean = df["sum"] / count
    denom = (count - 1).clip(lower=1)
    var = (df["sumsq"] - count * mean**2) / denom
    var = var.clip(lower=0).where(count > 1, 0.0)

    return pd.DataFrame(
        {
            "userId": df["userId"],
            "total_ratings": count.astype("int32"),
            "avg_rating": mean.astype("float32"),
            "rating_std": np.sqrt(var).astype("float32"),
            "min_rating": df["min"].astype("float32"),
            "max_rating": df["max"].astype("float32"),
            "avg_reward": (df["reward_sum"] / count).astype("float32"),
        }
    )


def build_recent_movie_columns(user_ids, recent_hist):
    rows = []
    for uid in user_ids:
        movie_list = [mid for _, mid in recent_hist[uid]]
        movie_list = movie_list + [None] * (RECENT_HISTORY_LEN - len(movie_list))
        rows.append(movie_list)
    cols = [f"recent_movie_{i + 1}" for i in range(RECENT_HISTORY_LEN)]
    recent_df = pd.DataFrame(rows, columns=cols)
    for c in cols:
        recent_df[c] = recent_df[c].astype("Int32")
    recent_df.insert(0, "userId", user_ids)
    return recent_df


# ------------------------------------------------------------------
# Pass 2: stream ratings_clean.csv again, merge the (tiny) user_segment
# lookup onto each chunk, and write rl_features.parquet incrementally.
# ------------------------------------------------------------------
def run_pass_two(user_segment_lookup):
    rl_path = FEATURES_DIR / "rl_features.parquet"
    rl_path.unlink(missing_ok=True)

    writer = None
    n_chunks = 0
    rl_rows = 0

    for chunk in ratings_chunks():
        n_chunks += 1

        rl_chunk = chunk[["userId", "movieId", "rating", "timestamp"]].copy()
        rl_chunk["reward"] = (rl_chunk["rating"] >= RATING_POSITIVE_THRESHOLD).astype("int8")
        rl_chunk = rl_chunk.merge(user_segment_lookup, on="userId", how="left")
        rl_chunk["user_segment"] = rl_chunk["user_segment"].astype("category")

        table = pa.Table.from_pandas(rl_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(rl_path, table.schema)
        writer.write_table(table)
        rl_rows += len(rl_chunk)

        print(f"  Pass 2 - chunk {n_chunks}: {rl_rows:,} rows written so far")

    if writer is not None:
        writer.close()

    return rl_rows


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    forecasting_path = FEATURES_DIR / "forecasting_features.csv"
    if forecasting_path.exists():
        print(f"Skipping forecasting features -- {forecasting_path} already exists, left as-is.")
    else:
        print(f"Warning: {forecasting_path} not found. This run does not regenerate it.")

    movies, genre_pairs = load_movie_genres()
    print(f"movies_clean: {movies.shape}")

    print("\n" + "=" * 70)
    print("GENOME EMBEDDINGS (chunked SVD compression, 1,128 -> 50 dims)")
    print("=" * 70)
    genome_lookup, genome_cols = build_genome_embeddings()
    show("genome_embeddings", genome_lookup)

    print("\n" + "=" * 70)
    print("PASS 1 - streaming aggregation (user/movie stats, favorite genre, recent history)")
    print("=" * 70)
    user_stats, movie_stats, user_genre_counts, recent_hist = run_pass_one(genre_pairs)

    favorite_idx = user_genre_counts.groupby("userId")["count"].idxmax()
    favorite_genre = user_genre_counts.loc[favorite_idx, ["userId", "genre"]].rename(
        columns={"genre": "favorite_genre"}
    )

    user_ids = list(recent_hist.keys())
    recent_df = build_recent_movie_columns(user_ids, recent_hist)

    print("\n" + "=" * 70)
    print("2. USER FEATURES")
    print("=" * 70)
    user_features = finalize_user_stats(user_stats)
    user_features["user_segment"] = user_features["total_ratings"].apply(segment_label).astype("category")
    user_features = user_features.merge(favorite_genre, on="userId", how="left")
    user_features = user_features.merge(recent_df, on="userId", how="left")
    user_features = user_features[
        [
            "userId", "total_ratings", "avg_rating", "rating_std", "min_rating", "max_rating",
            "favorite_genre", "user_segment",
            "recent_movie_1", "recent_movie_2", "recent_movie_3", "recent_movie_4", "recent_movie_5",
            "avg_reward",
        ]
    ]
    show("user_features", user_features)
    user_features_path = FEATURES_DIR / "user_features.parquet"
    user_features.to_parquet(user_features_path, index=False)
    print(f"Saved -> {user_features_path}")

    print("\nUser segment counts:")
    print(user_features["user_segment"].value_counts())

    print("\n" + "=" * 70)
    print("3. MOVIE FEATURES")
    print("=" * 70)
    movie_features = finalize_movie_stats(movie_stats)
    movie_features = movie_features.merge(movies[["movieId", "genres"]], on="movieId", how="left")
    movie_features = movie_features.merge(genome_lookup, on="movieId", how="left")
    movie_features[genome_cols] = movie_features[genome_cols].fillna(0.0)
    movie_features = movie_features[["movieId", "total_ratings", "avg_rating", "rating_std", "genres"] + genome_cols]
    show("movie_features", movie_features)
    movie_features_path = FEATURES_DIR / "movie_features.parquet"
    movie_features.to_parquet(movie_features_path, index=False)
    print(f"Saved -> {movie_features_path}")

    print("\n" + "=" * 70)
    print("PASS 2 - streaming rl_features.parquet to disk")
    print("=" * 70)
    user_segment_lookup = user_features[["userId", "user_segment"]]
    rl_rows = run_pass_two(user_segment_lookup)
    rl_features_path = FEATURES_DIR / "rl_features.parquet"
    print(f"Saved -> {rl_features_path} ({rl_rows:,} rows)")

    print("\n" + "=" * 70)
    print("6. OUTPUT FILE REPORT")
    print("=" * 70)
    print(f"\n{forecasting_path.name} (unchanged, left as-is)")
    print(f"  File size on disk: {forecasting_path.stat().st_size / 1e6:,.1f} MB")

    report_output_file(user_features_path, user_features)
    report_output_file(movie_features_path, movie_features)
    report_output_file(rl_features_path)


if __name__ == "__main__":
    main()
