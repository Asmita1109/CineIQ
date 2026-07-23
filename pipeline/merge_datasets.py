"""Merge the raw MovieLens and TMDB datasets into one combined dataset.

Standardizes TMDB's movies/ratings tables onto the exact MovieLens schema,
remaps TMDB's userId/movieId space so it can't collide with MovieLens's,
flags likely title+release-year duplicates between the two sources (without
removing them -- preprocess.py decides what to do with the flag), and writes
out the combined tables.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ML_DIR = RAW_DIR / "ml-25m"

RATING_SCALE_MIN = 0.5
RATING_SCALE_MAX = 5.0

ML_DTYPES = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}


def load_movielens():
    ratings = pd.read_csv(ML_DIR / "ratings.csv", dtype=ML_DTYPES)
    movies = pd.read_csv(ML_DIR / "movies.csv")
    return ratings, movies


def load_tmdb():
    movies = pd.read_csv(RAW_DIR / "tmdb_movies.csv")
    ratings = pd.read_csv(RAW_DIR / "tmdb_ratings.csv", dtype=ML_DTYPES)
    return ratings, movies


def build_id_map(unique_ids, start_at):
    """Sequential remap so the new ids start right after the MovieLens max."""
    return {old: start_at + i for i, old in enumerate(sorted(unique_ids))}


def standardize_tmdb(tmdb_movies, tmdb_ratings, ml_movies, ml_ratings):
    max_ml_movie_id = int(max(ml_movies["movieId"].max(), ml_ratings["movieId"].max()))
    max_ml_user_id = int(ml_ratings["userId"].max())

    movie_id_map = build_id_map(tmdb_movies["movieId"].unique(), max_ml_movie_id + 1)
    user_id_map = build_id_map(tmdb_ratings["userId"].unique(), max_ml_user_id + 1)

    tmdb_movies_std = tmdb_movies[["movieId", "title", "genres", "release_date"]].copy()
    tmdb_movies_std["movieId"] = tmdb_movies_std["movieId"].map(movie_id_map).astype("int32")

    tmdb_ratings_std = tmdb_ratings[["userId", "movieId", "rating", "timestamp"]].copy()
    tmdb_ratings_std["userId"] = tmdb_ratings_std["userId"].map(user_id_map).astype("int32")
    tmdb_ratings_std["movieId"] = tmdb_ratings_std["movieId"].map(movie_id_map).astype("int32")
    tmdb_ratings_std["rating"] = tmdb_ratings_std["rating"].clip(RATING_SCALE_MIN, RATING_SCALE_MAX)

    return tmdb_movies_std, tmdb_ratings_std, movie_id_map, user_id_map


def flag_title_year_duplicates(ml_movies, tmdb_movies_std):
    """Flag movies that share a normalized title + release year across sources.
    Nothing is removed here -- preprocess.py decides what to do with the flag."""
    ml_movies = ml_movies.copy()
    tmdb_movies_std = tmdb_movies_std.copy()

    ml_movies["year"] = ml_movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("Int64")
    ml_movies["title_norm"] = (
        ml_movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip().str.lower()
    )

    tmdb_movies_std["year"] = pd.to_datetime(
        tmdb_movies_std["release_date"], errors="coerce"
    ).dt.year.astype("Int64")
    tmdb_movies_std["title_norm"] = tmdb_movies_std["title"].str.strip().str.lower()

    ml_keys = ml_movies.loc[ml_movies["year"].notna(), ["title_norm", "year"]].drop_duplicates()
    tmdb_keys = tmdb_movies_std.loc[
        tmdb_movies_std["year"].notna(), ["title_norm", "year"]
    ].drop_duplicates()
    dup_keys = ml_keys.merge(tmdb_keys, on=["title_norm", "year"], how="inner")
    dup_keys["is_duplicate_title_year"] = True

    ml_movies = ml_movies.merge(dup_keys, on=["title_norm", "year"], how="left")
    ml_movies["is_duplicate_title_year"] = ml_movies["is_duplicate_title_year"].fillna(False).astype(bool)

    tmdb_movies_std = tmdb_movies_std.merge(dup_keys, on=["title_norm", "year"], how="left")
    tmdb_movies_std["is_duplicate_title_year"] = (
        tmdb_movies_std["is_duplicate_title_year"].fillna(False).astype(bool)
    )

    return ml_movies, tmdb_movies_std, len(dup_keys)


def date_range(ratings):
    dates = pd.to_datetime(ratings["timestamp"], unit="s")
    return dates.min(), dates.max()


def print_report(ml_ratings, ml_movies, tmdb_ratings_std, tmdb_movies_std, combined_ratings, combined_movies, n_dup_keys):
    print("\n" + "=" * 70)
    print("BEFORE / AFTER REPORT")
    print("=" * 70)

    ml_rmin, ml_rmax = date_range(ml_ratings)
    tmdb_rmin, tmdb_rmax = date_range(tmdb_ratings_std)
    combined_rmin, combined_rmax = date_range(combined_ratings)

    print("\nBEFORE (raw sources)")
    print(f"  ml-25m ratings.csv:    {len(ml_ratings):>12,} rows   dates {ml_rmin.date()} to {ml_rmax.date()}")
    print(f"  ml-25m movies.csv:     {len(ml_movies):>12,} rows")
    print(f"  tmdb_ratings.csv:      {len(tmdb_ratings_std):>12,} rows   dates {tmdb_rmin.date()} to {tmdb_rmax.date()}")
    print(f"  tmdb_movies.csv:       {len(tmdb_movies_std):>12,} rows")

    print("\nAFTER (combined)")
    print(f"  combined_ratings.csv:  {len(combined_ratings):>12,} rows   dates {combined_rmin.date()} to {combined_rmax.date()}")
    print(f"  combined_movies.csv:   {len(combined_movies):>12,} rows")

    print("\nSOURCE BREAKDOWN")
    print("  combined_ratings.csv:")
    print(combined_ratings["source"].value_counts().to_string())
    print("  combined_movies.csv:")
    print(combined_movies["source"].value_counts().to_string())

    print("\nDUPLICATE FLAGGING (by normalized title + release year)")
    print(f"  Matching title+year keys found across both sources: {n_dup_keys:,}")
    print(f"  Flagged MovieLens rows:  {int(ml_movies['is_duplicate_title_year'].sum()):,}")
    print(f"  Flagged TMDB rows:       {int(tmdb_movies_std['is_duplicate_title_year'].sum()):,}")
    print("  (flagged only -- not removed; preprocess.py decides how to handle them)")


def main():
    print("Loading MovieLens (ml-25m)...")
    ml_ratings, ml_movies = load_movielens()
    print(f"  ratings.csv: {ml_ratings.shape}")
    print(f"  movies.csv:  {ml_movies.shape}")

    print("\nLoading TMDB...")
    tmdb_ratings, tmdb_movies = load_tmdb()
    print(f"  tmdb_ratings.csv: {tmdb_ratings.shape}")
    print(f"  tmdb_movies.csv:  {tmdb_movies.shape}")

    print("\nStandardizing TMDB onto the MovieLens schema and remapping ids...")
    tmdb_movies_std, tmdb_ratings_std, movie_id_map, user_id_map = standardize_tmdb(
        tmdb_movies, tmdb_ratings, ml_movies, ml_ratings
    )
    print(f"  TMDB movieId range remapped to: {min(movie_id_map.values()):,} - {max(movie_id_map.values()):,}")
    print(f"  TMDB userId range remapped to:  {min(user_id_map.values()):,} - {max(user_id_map.values()):,}")

    print("\nFlagging title+year duplicates between sources...")
    ml_movies_flagged, tmdb_movies_flagged, n_dup_keys = flag_title_year_duplicates(ml_movies, tmdb_movies_std)
    print(f"  {n_dup_keys:,} overlapping (title, year) keys found")

    ml_movies_final = ml_movies_flagged[["movieId", "title", "genres", "is_duplicate_title_year"]].copy()
    ml_movies_final["source"] = "movielens"

    tmdb_movies_final = tmdb_movies_flagged[["movieId", "title", "genres", "is_duplicate_title_year"]].copy()
    tmdb_movies_final["source"] = "tmdb"

    combined_movies = pd.concat([ml_movies_final, tmdb_movies_final], ignore_index=True)

    ml_ratings_final = ml_ratings.copy()
    ml_ratings_final["source"] = "movielens"

    tmdb_ratings_final = tmdb_ratings_std.copy()
    tmdb_ratings_final["source"] = "tmdb"

    combined_ratings = pd.concat([ml_ratings_final, tmdb_ratings_final], ignore_index=True)

    print_report(
        ml_ratings,
        ml_movies_flagged,
        tmdb_ratings_std,
        tmdb_movies_flagged,
        combined_ratings,
        combined_movies,
        n_dup_keys,
    )

    ratings_path = RAW_DIR / "combined_ratings.csv"
    movies_path = RAW_DIR / "combined_movies.csv"
    combined_ratings.to_csv(ratings_path, index=False)
    combined_movies.to_csv(movies_path, index=False)

    print(f"\nSaved -> {ratings_path}  ({combined_ratings.shape[0]:,} rows)")
    print(combined_ratings.head())
    print(f"\nSaved -> {movies_path}  ({combined_movies.shape[0]:,} rows)")
    print(combined_movies.head())


if __name__ == "__main__":
    main()
