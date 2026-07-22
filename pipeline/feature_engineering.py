"""Build feature tables for the forecasting, recommendation, and RL components.

Processes ratings_clean.csv in chunks (CHUNK_SIZE rows at a time) so the
pipeline never holds the full ratings table in memory. Aggregates (user/movie
stats, genre-month counts, favorite genre, recent history) are accumulated
chunk-by-chunk in a first streaming pass and combined at the end. A second
streaming pass then merges each chunk against those small, precomputed lookup
tables and appends the row-level feature tables straight to disk.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

CHUNK_SIZE = 500_000
RATING_POSITIVE_THRESHOLD = 4.0
CASUAL_MAX = 20
REGULAR_MAX = 100

RATINGS_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
    "year": "int16",
    "month": "int8",
}


def ratings_chunks():
    return pd.read_csv(
        PROCESSED_DIR / "ratings_clean.csv", dtype=RATINGS_DTYPES, chunksize=CHUNK_SIZE
    )


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


# ------------------------------------------------------------------
# Pass 1: stream ratings_clean.csv once, accumulate every aggregate
# ------------------------------------------------------------------
def run_pass_one(genre_pairs):
    user_partials = []
    movie_partials = []
    user_genre_partials = []
    genre_month_partials = []
    recent_hist = {}

    n_chunks = 0
    n_rows = 0
    for chunk in ratings_chunks():
        n_chunks += 1
        n_rows += len(chunk)
        chunk = chunk.copy()
        chunk["rating_sq"] = chunk["rating"].astype("float64") ** 2

        user_partials.append(
            chunk.groupby("userId").agg(
                count=("rating", "size"), sum=("rating", "sum"), sumsq=("rating_sq", "sum")
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
        genre_month_partials.append(
            chunk_genre.groupby(["genre", "year", "month"])
            .agg(count=("rating", "size"), sum=("rating", "sum"))
            .reset_index()
        )

        top5_chunk = (
            chunk.sort_values("timestamp", ascending=False)
            .groupby("userId")
            .head(5)[["userId", "timestamp", "movieId"]]
        )
        for uid, ts, mid in zip(
            top5_chunk["userId"].to_numpy(),
            top5_chunk["timestamp"].to_numpy(),
            top5_chunk["movieId"].to_numpy(),
        ):
            lst = recent_hist.setdefault(int(uid), [])
            lst.append((int(ts), int(mid)))
            if len(lst) > 5:
                lst.sort(key=lambda x: x[0], reverse=True)
                del lst[5:]

        print(f"  Pass 1 - chunk {n_chunks}: {n_rows:,} rows processed so far")

    user_stats = pd.concat(user_partials).groupby("userId").sum().reset_index()
    movie_stats = pd.concat(movie_partials).groupby("movieId").sum().reset_index()
    user_genre_counts = (
        pd.concat(user_genre_partials).groupby(["userId", "genre"])["count"].sum().reset_index()
    )
    genre_month_counts = (
        pd.concat(genre_month_partials)
        .groupby(["genre", "year", "month"])
        .sum(numeric_only=True)
        .reset_index()
    )

    for lst in recent_hist.values():
        lst.sort(key=lambda x: x[0], reverse=True)

    return user_stats, movie_stats, user_genre_counts, genre_month_counts, recent_hist


def finalize_stats(raw, id_col, prefix):
    df = raw.copy()
    count = df["count"]
    mean = df["sum"] / count
    denom = (count - 1).clip(lower=1)
    var = (df["sumsq"] - count * mean**2) / denom
    var = var.clip(lower=0).where(count > 1, 0.0)

    out = pd.DataFrame(
        {
            id_col: df[id_col],
            f"{prefix}_total_ratings": count.astype("int32"),
            f"{prefix}_avg_rating": mean.astype("float32"),
            f"{prefix}_rating_std": np.sqrt(var).astype("float32"),
        }
    )
    return out


def build_forecasting_features(genre_month_counts):
    genre_month = genre_month_counts.rename(columns={"count": "rating_count", "sum": "rating_sum"})
    genre_month["avg_rating"] = genre_month["rating_sum"] / genre_month["rating_count"]
    genre_month["period"] = pd.PeriodIndex.from_fields(
        year=genre_month["year"], month=genre_month["month"], freq="M"
    )

    all_periods = pd.period_range(genre_month["period"].min(), genre_month["period"].max(), freq="M")
    genres = sorted(genre_month["genre"].unique())
    full_index = pd.MultiIndex.from_product([genres, all_periods], names=["genre", "period"])

    genre_month_full = (
        genre_month.set_index(["genre", "period"])[["rating_count", "avg_rating"]]
        .reindex(full_index)
        .reset_index()
    )
    genre_month_full["rating_count"] = genre_month_full["rating_count"].fillna(0).astype("int32")
    genre_month_full["year"] = genre_month_full["period"].dt.year.astype("int16")
    genre_month_full["month"] = genre_month_full["period"].dt.month.astype("int8")
    genre_month_full = genre_month_full.sort_values(["genre", "period"]).reset_index(drop=True)

    counts = genre_month_full.groupby("genre")["rating_count"]
    genre_month_full["lag_1"] = counts.shift(1)
    genre_month_full["lag_2"] = counts.shift(2)
    genre_month_full["lag_3"] = counts.shift(3)
    genre_month_full["rolling_3month_avg"] = counts.transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    genre_month_full["rolling_6month_avg"] = counts.transform(
        lambda s: s.rolling(window=6, min_periods=1).mean()
    )

    return genre_month_full[
        [
            "genre",
            "year",
            "month",
            "rating_count",
            "avg_rating",
            "lag_1",
            "lag_2",
            "lag_3",
            "rolling_3month_avg",
            "rolling_6month_avg",
        ]
    ]


# ------------------------------------------------------------------
# Pass 2: stream ratings_clean.csv again, merge against small lookup
# tables built in pass 1, and append each chunk straight to disk.
# ------------------------------------------------------------------
def run_pass_two(user_agg, movie_agg, user_segment_map, recent_movie_ids):
    rec_path = FEATURES_DIR / "rec_features.csv"
    rl_path = FEATURES_DIR / "rl_features.csv"
    rec_path.unlink(missing_ok=True)
    rl_path.unlink(missing_ok=True)

    header_written = False
    n_chunks = 0
    rec_rows = 0
    rl_rows = 0

    for chunk in ratings_chunks():
        n_chunks += 1

        rec_chunk = chunk[["userId", "movieId", "rating"]].merge(user_agg, on="userId", how="left")
        rec_chunk = rec_chunk.merge(movie_agg, on="movieId", how="left")
        rec_chunk.to_csv(rec_path, mode="a", header=not header_written, index=False)
        rec_rows += len(rec_chunk)

        rl_chunk = chunk[["userId", "movieId", "rating", "year", "month"]].copy()
        rl_chunk["reward"] = (rl_chunk["rating"] >= RATING_POSITIVE_THRESHOLD).astype("int8")
        rl_chunk["user_segment"] = rl_chunk["userId"].map(user_segment_map)
        rl_chunk["recent_movie_ids"] = rl_chunk["userId"].map(recent_movie_ids)
        rl_chunk.to_csv(rl_path, mode="a", header=not header_written, index=False)
        rl_rows += len(rl_chunk)

        header_written = True
        print(f"  Pass 2 - chunk {n_chunks}: {rec_rows:,} rows written so far")

    return rec_rows, rl_rows


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Streaming ratings_clean.csv in chunks of {CHUNK_SIZE:,} rows")
    movies, genre_pairs = load_movie_genres()
    print(f"movies_clean: {movies.shape}")

    print("\n" + "=" * 70)
    print("PASS 1 - streaming aggregation (user/movie stats, genre-month, recent history)")
    print("=" * 70)
    user_stats, movie_stats, user_genre_counts, genre_month_counts, recent_hist = run_pass_one(
        genre_pairs
    )

    user_agg = finalize_stats(user_stats, "userId", "user")
    movie_agg = finalize_stats(movie_stats, "movieId", "movie")

    favorite_idx = user_genre_counts.groupby("userId")["count"].idxmax()
    favorite_genre = user_genre_counts.loc[favorite_idx, ["userId", "genre"]].rename(
        columns={"genre": "user_favorite_genre"}
    )
    user_agg = user_agg.merge(favorite_genre, on="userId", how="left")

    movie_agg = movie_agg.merge(movies[["movieId", "genres"]], on="movieId", how="left")
    movie_agg = movie_agg.rename(columns={"genres": "movie_genres"})

    user_segment = user_agg[["userId", "user_total_ratings"]].copy()
    user_segment["user_segment"] = user_segment["user_total_ratings"].apply(segment_label)
    user_segment_map = user_segment.set_index("userId")["user_segment"]

    recent_movie_ids = {
        uid: "|".join(str(mid) for _, mid in lst) for uid, lst in recent_hist.items()
    }

    print("\n" + "=" * 70)
    print("1. FORECASTING FEATURES")
    print("=" * 70)
    forecasting_features = build_forecasting_features(genre_month_counts)
    show("forecasting_features", forecasting_features)
    forecasting_features.to_csv(FEATURES_DIR / "forecasting_features.csv", index=False)
    print(f"Saved -> {FEATURES_DIR / 'forecasting_features.csv'}")

    print("\n" + "=" * 70)
    print("2/3. RECOMMENDATION + RL LOOKUP TABLES")
    print("=" * 70)
    show("user_features", user_agg)
    show("movie_features", movie_agg)
    print("\nUser segment counts:")
    print(user_segment["user_segment"].value_counts())

    print("\n" + "=" * 70)
    print("PASS 2 - streaming rec_features.csv / rl_features.csv to disk")
    print("=" * 70)
    rec_rows, rl_rows = run_pass_two(user_agg, movie_agg, user_segment_map, recent_movie_ids)
    print(f"\nSaved -> {FEATURES_DIR / 'rec_features.csv'} ({rec_rows:,} rows)")
    print(f"Saved -> {FEATURES_DIR / 'rl_features.csv'} ({rl_rows:,} rows)")

    show("rec_features (sample)", pd.read_csv(FEATURES_DIR / "rec_features.csv", nrows=5))
    show("rl_features (sample)", pd.read_csv(FEATURES_DIR / "rl_features.csv", nrows=5))


if __name__ == "__main__":
    main()
