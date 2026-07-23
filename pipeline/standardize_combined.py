"""Standardize combined_movies.csv / combined_ratings.csv into one dataset
with no trace of which rows came from which source.

Formats TMDB titles as "Movie Name (YYYY)" (re-deriving the release year
from the original tmdb_movies.csv, joined back via the deterministic
movieId remap merge_datasets.py applied), strips every source/provenance
column, then redoes duplicate detection on the now-consistent title format
and merges duplicate TMDB movies onto their canonical MovieLens movieId
instead of just dropping them.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

RATING_SCALE_MIN = 0.5
RATING_SCALE_MAX = 5.0

MOVIE_DTYPES = {"movieId": "int32", "title": "object", "genres": "object"}
RATING_DTYPES = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}


def recover_tmdb_release_dates(combined_movies, max_ml_movie_id):
    """combined_movies' TMDB movieIds are a deterministic sequential remap of
    sorted(original tmdb_movies.csv movieId) starting at max_ml_movie_id + 1
    (see merge_datasets.py:build_id_map). Invert that mapping to pull each
    TMDB row's original release_date back in."""
    tmdb_raw = pd.read_csv(RAW_DIR / "tmdb_movies.csv", usecols=["movieId", "release_date"])
    old_ids_sorted = sorted(tmdb_raw["movieId"].unique())
    new_ids_sorted = range(max_ml_movie_id + 1, max_ml_movie_id + 1 + len(old_ids_sorted))
    new_to_old = dict(zip(new_ids_sorted, old_ids_sorted))

    tmdb_rows = combined_movies[combined_movies["movieId"] > max_ml_movie_id].copy()
    tmdb_rows["_old_movieId"] = tmdb_rows["movieId"].map(new_to_old)
    tmdb_rows = tmdb_rows.merge(
        tmdb_raw.rename(columns={"movieId": "_old_movieId"}), on="_old_movieId", how="left"
    )
    return tmdb_rows


def main():
    print("Loading combined_movies.csv / combined_ratings.csv...")
    movies = pd.read_csv(RAW_DIR / "combined_movies.csv")
    ratings = pd.read_csv(RAW_DIR / "combined_ratings.csv", dtype=RATING_DTYPES)
    print(f"  combined_movies.csv:  {movies.shape}")
    print(f"  combined_ratings.csv: {ratings.shape}")

    before_movies = len(movies)
    before_ratings = len(ratings)

    max_ml_movie_id = int(movies.loc[movies["source"] == "movielens", "movieId"].max())
    max_ml_user_id = int(ratings.loc[ratings["source"] == "movielens", "userId"].max())
    print(f"\nMovieLens movieId max: {max_ml_movie_id:,}  (TMDB movieIds start at {max_ml_movie_id + 1:,})")
    print(f"MovieLens userId max:  {max_ml_user_id:,}  (TMDB userIds start at {max_ml_user_id + 1:,})")

    # ------------------------------------------------------------------
    # 1. Movies standardization
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. MOVIES STANDARDIZATION")
    print("=" * 70)

    ml_movies = movies.loc[movies["source"] == "movielens", ["movieId", "title", "genres"]].copy()

    tmdb_rows = recover_tmdb_release_dates(movies, max_ml_movie_id)
    tmdb_year = pd.to_datetime(tmdb_rows["release_date"], errors="coerce").dt.year
    has_year = tmdb_year.notna()
    tmdb_rows.loc[has_year, "title"] = (
        tmdb_rows.loc[has_year, "title"].astype(str).str.strip()
        + " ("
        + tmdb_year[has_year].astype(int).astype(str)
        + ")"
    )
    print(f"  TMDB rows with a usable release_date: {int(has_year.sum()):,} / {len(tmdb_rows):,}")
    print(f"  TMDB rows with missing/invalid release_date (title left unchanged): {int((~has_year).sum()):,}")

    tmdb_movies_std = tmdb_rows[["movieId", "title", "genres"]].copy()

    unified_movies = pd.concat([ml_movies, tmdb_movies_std], ignore_index=True)
    unified_movies["movieId"] = unified_movies["movieId"].astype("int32")
    print(f"\n  Unified movies (pre-dedup): {len(unified_movies):,} rows")
    print(f"  Columns: {list(unified_movies.columns)}")

    # ------------------------------------------------------------------
    # 2. Ratings standardization
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. RATINGS STANDARDIZATION")
    print("=" * 70)
    unified_ratings = ratings[["userId", "movieId", "rating", "timestamp"]].copy()
    rmin, rmax = float(unified_ratings["rating"].min()), float(unified_ratings["rating"].max())
    out_of_range = int(
        ((unified_ratings["rating"] < RATING_SCALE_MIN) | (unified_ratings["rating"] > RATING_SCALE_MAX)).sum()
    )
    print(f"  Rating range across all rows: {rmin} - {rmax}")
    print(f"  Rows outside {RATING_SCALE_MIN}-{RATING_SCALE_MAX}: {out_of_range:,}")
    print(f"  Columns: {list(unified_ratings.columns)}")

    # ------------------------------------------------------------------
    # 3. Redo duplicate detection on the standardized "Title (YYYY)" format
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. DUPLICATE DETECTION (standardized title format)")
    print("=" * 70)

    extracted = unified_movies["title"].str.extract(r"^(.*?)\s*\((\d{4})\)\s*$")
    unified_movies["year"] = extracted[1].astype("Int64")
    unified_movies["title_norm"] = (
        extracted[0]
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    unified_movies["is_tmdb"] = unified_movies["movieId"] > max_ml_movie_id

    ml_keys = unified_movies.loc[~unified_movies["is_tmdb"], ["movieId", "title_norm", "year"]].dropna(
        subset=["title_norm", "year"]
    )
    tmdb_keys = unified_movies.loc[unified_movies["is_tmdb"], ["movieId", "title_norm", "year"]].dropna(
        subset=["title_norm", "year"]
    )

    # one canonical MovieLens movieId per (title_norm, year) key
    ml_canonical = (
        ml_keys.groupby(["title_norm", "year"])["movieId"]
        .min()
        .reset_index()
        .rename(columns={"movieId": "canonical_movieId"})
    )

    dup_map = tmdb_keys.merge(ml_canonical, on=["title_norm", "year"], how="inner")
    tmdb_to_ml = dict(zip(dup_map["movieId"], dup_map["canonical_movieId"]))
    n_dup_keys = dup_map[["title_norm", "year"]].drop_duplicates().shape[0]
    print(f"  Duplicate (title, year) keys matched: {n_dup_keys:,}")
    print(f"  TMDB movie rows to remap/drop: {len(dup_map):,}")

    remap_mask = unified_ratings["movieId"].isin(tmdb_to_ml)
    n_ratings_remapped = int(remap_mask.sum())
    remapped_ids = unified_ratings.loc[remap_mask, "movieId"].map(tmdb_to_ml).astype("int32")
    unified_ratings.loc[remap_mask, "movieId"] = remapped_ids
    print(f"  Ratings remapped onto their canonical MovieLens movieId: {n_ratings_remapped:,}")

    unified_movies = unified_movies.loc[~unified_movies["movieId"].isin(tmdb_to_ml)].copy()
    unified_movies = unified_movies.drop(columns=["year", "title_norm", "is_tmdb"])
    unified_movies["movieId"] = unified_movies["movieId"].astype("int32")
    print(f"  Duplicate movie rows dropped (MovieLens version kept as canonical): {len(dup_map):,}")

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    movies_path = RAW_DIR / "combined_movies.csv"
    ratings_path = RAW_DIR / "combined_ratings.csv"
    unified_movies.to_csv(movies_path, index=False)
    unified_ratings.to_csv(ratings_path, index=False)

    # ------------------------------------------------------------------
    # 5. Final report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. FINAL REPORT")
    print("=" * 70)
    print(
        f"\nmovies:  {before_movies:,} -> {len(unified_movies):,}  "
        f"({before_movies - len(unified_movies):,} duplicate TMDB rows dropped)"
    )
    print(
        f"ratings: {before_ratings:,} -> {len(unified_ratings):,}  "
        f"(row count unchanged; {n_ratings_remapped:,} rows remapped onto a canonical movieId)"
    )

    print(f"\nSample MovieLens-origin rows (movieId <= {max_ml_movie_id:,}):")
    print(unified_movies[unified_movies["movieId"] <= max_ml_movie_id].head(5).to_string(index=False))

    print(f"\nSample TMDB-origin rows (movieId > {max_ml_movie_id:,}) -- confirming identical 'Title (YYYY)' format:")
    print(unified_movies[unified_movies["movieId"] > max_ml_movie_id].head(5).to_string(index=False))

    print(f"\nFinal duplicate count handled: {len(dup_map):,} TMDB movies merged into their MovieLens canonical entry")
    print(f"\nSaved -> {movies_path}  {unified_movies.shape}")
    print(f"Saved -> {ratings_path}  {unified_ratings.shape}")


if __name__ == "__main__":
    main()
