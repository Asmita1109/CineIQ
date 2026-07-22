"""Download and ingest the MovieLens 25M dataset."""

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
ZIP_PATH = DATA_DIR / "ml-25m.zip"
EXTRACTED_DIR = RAW_DIR / "ml-25m"


def download_dataset():
    if ZIP_PATH.exists():
        print(f"Zip already exists at {ZIP_PATH}, skipping download.")
        return
    print(f"Downloading {DATASET_URL} ...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urlretrieve(DATASET_URL, ZIP_PATH)
    print(f"Downloaded to {ZIP_PATH}")


def extract_dataset():
    if EXTRACTED_DIR.exists():
        print(f"Dataset already extracted at {EXTRACTED_DIR}, skipping extraction.")
        return
    print(f"Extracting {ZIP_PATH} to {RAW_DIR} ...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(RAW_DIR)
    print(f"Extracted to {EXTRACTED_DIR}")


def load_dataframes():
    ratings = pd.read_csv(EXTRACTED_DIR / "ratings.csv")
    movies = pd.read_csv(EXTRACTED_DIR / "movies.csv")
    tags = pd.read_csv(EXTRACTED_DIR / "tags.csv")

    for name, df in [("ratings", ratings), ("movies", movies), ("tags", tags)]:
        print(f"\n{name}.csv -- shape: {df.shape}")
        print(df.head())

    return ratings, movies, tags


def write_summary(ratings, movies, tags):
    min_date = pd.to_datetime(ratings["timestamp"], unit="s").min()
    max_date = pd.to_datetime(ratings["timestamp"], unit="s").max()

    lines = [
        "MovieLens 25M -- Summary Statistics",
        "=" * 40,
        "",
        "ratings.csv",
        f"  rows: {len(ratings)}",
        f"  columns: {list(ratings.columns)}",
        "",
        "movies.csv",
        f"  rows: {len(movies)}",
        f"  columns: {list(movies.columns)}",
        "",
        "tags.csv",
        f"  rows: {len(tags)}",
        f"  columns: {list(tags.columns)}",
        "",
        f"Ratings date range: {min_date.date()} to {max_date.date()}",
        f"Unique users: {ratings['userId'].nunique()}",
        f"Unique movies: {ratings['movieId'].nunique()}",
    ]

    summary_path = RAW_DIR / "summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary written to {summary_path}")


def main():
    download_dataset()
    extract_dataset()
    ratings, movies, tags = load_dataframes()
    write_summary(ratings, movies, tags)


if __name__ == "__main__":
    main()
