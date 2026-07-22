"""Clean and preprocess the raw MovieLens 25M dataset."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "ml-25m"
PROCESSED_DIR = DATA_DIR / "processed"

MIN_RATINGS_PER_USER = 5
MIN_RATINGS_PER_MOVIE = 5
VALID_RATING_MIN = 0.5
VALID_RATING_MAX = 5.0


def load_raw():
    ratings = pd.read_csv(
        RAW_DIR / "ratings.csv",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )
    movies = pd.read_csv(RAW_DIR / "movies.csv")
    tags = pd.read_csv(
        RAW_DIR / "tags.csv",
        dtype={"userId": "int32", "movieId": "int32", "timestamp": "int64"},
    )
    return ratings, movies, tags


def report_missing(df, name):
    missing = df.isna().sum()
    total = int(missing.sum())
    print(f"  Missing values in {name}:")
    if total == 0:
        print("    none")
    else:
        for col, count in missing[missing > 0].items():
            print(f"    {col}: {count:,}")


def report_quality(ratings, movies, tags):
    print("=" * 70)
    print("DATA QUALITY REPORT (before cleaning)")
    print("=" * 70)

    print(f"\nratings.csv: {len(ratings):,} rows")
    report_missing(ratings, "ratings")
    dup_ratings = int(ratings.duplicated().sum())
    print(f"  Duplicate rows: {dup_ratings:,}")
    invalid_ratings = int(
        ((ratings["rating"] < VALID_RATING_MIN) | (ratings["rating"] > VALID_RATING_MAX)).sum()
    )
    print(f"  Invalid ratings (outside {VALID_RATING_MIN}-{VALID_RATING_MAX}): {invalid_ratings:,}")

    ratings_per_user = ratings.groupby("userId").size()
    users_below_min = int((ratings_per_user < MIN_RATINGS_PER_USER).sum())
    print(
        f"  Users with fewer than {MIN_RATINGS_PER_USER} ratings: {users_below_min:,} "
        f"(of {ratings_per_user.shape[0]:,} users)"
    )

    ratings_per_movie = ratings.groupby("movieId").size()
    movies_below_min = int((ratings_per_movie < MIN_RATINGS_PER_MOVIE).sum())
    print(
        f"  Movies with fewer than {MIN_RATINGS_PER_MOVIE} ratings: {movies_below_min:,} "
        f"(of {ratings_per_movie.shape[0]:,} rated movies)"
    )

    print(f"\nmovies.csv: {len(movies):,} rows")
    report_missing(movies, "movies")
    dup_movies = int(movies.duplicated().sum())
    print(f"  Duplicate rows: {dup_movies:,}")

    print(f"\ntags.csv: {len(tags):,} rows")
    report_missing(tags, "tags")
    dup_tags = int(tags.duplicated().sum())
    print(f"  Duplicate rows: {dup_tags:,}")


def clean_ratings(ratings):
    stats = {"start": len(ratings)}

    ratings = ratings.drop_duplicates()
    stats["after_dedup"] = len(ratings)

    ratings = ratings[
        (ratings["rating"] >= VALID_RATING_MIN) & (ratings["rating"] <= VALID_RATING_MAX)
    ]
    stats["after_invalid_removed"] = len(ratings)

    ratings_per_user = ratings.groupby("userId").size()
    valid_users = ratings_per_user[ratings_per_user >= MIN_RATINGS_PER_USER].index
    ratings = ratings[ratings["userId"].isin(valid_users)]
    stats["after_user_filter"] = len(ratings)

    ratings_per_movie = ratings.groupby("movieId").size()
    valid_movies = ratings_per_movie[ratings_per_movie >= MIN_RATINGS_PER_MOVIE].index
    ratings = ratings[ratings["movieId"].isin(valid_movies)]
    stats["after_movie_filter"] = len(ratings)

    ratings = ratings.copy()
    ratings["date"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["year"] = ratings["date"].dt.year.astype("int16")
    ratings["month"] = ratings["date"].dt.month.astype("int8")

    return ratings, stats


def clean_movies(movies):
    stats = {"start": len(movies)}

    movies = movies.drop_duplicates().copy()
    stats["after_dedup"] = len(movies)

    movies["genres_list"] = movies["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )

    return movies, stats


def clean_tags(tags):
    stats = {"start": len(tags)}

    tags = tags.drop_duplicates().copy()
    stats["after_dedup"] = len(tags)

    tags["date"] = pd.to_datetime(tags["timestamp"], unit="s")
    tags["year"] = tags["date"].dt.year.astype("int16")
    tags["month"] = tags["date"].dt.month.astype("int8")

    return tags, stats


def explode_genres(movies_clean):
    """One row per (movieId, genre) pair."""
    genre_pairs = movies_clean[["movieId", "genres_list"]].explode("genres_list")
    genre_pairs = genre_pairs.rename(columns={"genres_list": "genre"})
    genre_pairs = genre_pairs[genre_pairs["genre"].notna() & (genre_pairs["genre"] != "")]
    return genre_pairs.reset_index(drop=True)


def build_ratings_with_genres(ratings_clean, genre_pairs):
    merged = ratings_clean.merge(genre_pairs, on="movieId", how="inner")
    return merged[["userId", "movieId", "rating", "year", "month", "genre"]]


def print_before_after_report(ratings_stats, movies_stats, tags_stats, ratings_with_genres_len):
    print("\n" + "=" * 70)
    print("BEFORE / AFTER SUMMARY")
    print("=" * 70)

    r = ratings_stats
    dup_dropped = r["start"] - r["after_dedup"]
    invalid_dropped = r["after_dedup"] - r["after_invalid_removed"]
    user_filter_dropped = r["after_invalid_removed"] - r["after_user_filter"]
    movie_filter_dropped = r["after_user_filter"] - r["after_movie_filter"]
    total_dropped = r["start"] - r["after_movie_filter"]

    print("\nratings.csv")
    print(f"  Before:                                    {r['start']:,} rows")
    print(f"  - duplicates dropped:                      {dup_dropped:,}")
    print(f"  - invalid ratings dropped:                  {invalid_dropped:,}")
    print(f"  - rows from users  < {MIN_RATINGS_PER_USER} ratings dropped:      {user_filter_dropped:,}")
    print(f"  - rows from movies < {MIN_RATINGS_PER_MOVIE} ratings dropped:      {movie_filter_dropped:,}")
    print(f"  After:                                     {r['after_movie_filter']:,} rows")
    print(f"  Total dropped:                              {total_dropped:,} ({total_dropped / r['start'] * 100:.2f}%)")

    m = movies_stats
    print("\nmovies.csv")
    print(f"  Before: {m['start']:,} rows")
    print(f"  - duplicates dropped: {m['start'] - m['after_dedup']:,}")
    print(f"  After:  {m['after_dedup']:,} rows")

    t = tags_stats
    print("\ntags.csv")
    print(f"  Before: {t['start']:,} rows")
    print(f"  - duplicates dropped: {t['start'] - t['after_dedup']:,}")
    print(f"  After:  {t['after_dedup']:,} rows")

    print("\nratings_with_genres.csv")
    print(f"  Rows (ratings_clean exploded across movie genres): {ratings_with_genres_len:,}")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data from", RAW_DIR)
    ratings, movies, tags = load_raw()

    report_quality(ratings, movies, tags)

    print("\n" + "=" * 70)
    print("CLEANING")
    print("=" * 70)

    ratings_clean, ratings_stats = clean_ratings(ratings)
    movies_clean, movies_stats = clean_movies(movies)
    tags_clean, tags_stats = clean_tags(tags)
    genre_pairs = explode_genres(movies_clean)
    ratings_with_genres = build_ratings_with_genres(ratings_clean, genre_pairs)

    print("\nSaving cleaned files to", PROCESSED_DIR)
    ratings_clean.drop(columns=["date"]).to_csv(PROCESSED_DIR / "ratings_clean.csv", index=False)
    print(f"  Wrote ratings_clean.csv ({len(ratings_clean):,} rows)")

    movies_clean.to_csv(PROCESSED_DIR / "movies_clean.csv", index=False)
    print(f"  Wrote movies_clean.csv ({len(movies_clean):,} rows)")

    tags_clean.drop(columns=["date"]).to_csv(PROCESSED_DIR / "tags_clean.csv", index=False)
    print(f"  Wrote tags_clean.csv ({len(tags_clean):,} rows)")

    ratings_with_genres.to_csv(PROCESSED_DIR / "ratings_with_genres.csv", index=False)
    print(f"  Wrote ratings_with_genres.csv ({len(ratings_with_genres):,} rows)")

    print_before_after_report(ratings_stats, movies_stats, tags_stats, len(ratings_with_genres))


if __name__ == "__main__":
    main()
