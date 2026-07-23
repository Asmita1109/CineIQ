"""Audit the is_duplicate_title_year flag in data/raw/combined_movies.csv.

Pairs up the flagged MovieLens/TMDB rows, shows them side by side, and
classifies why each pair matched (exact title / casing / punctuation /
leading-article placement / other) so preprocess.py's dedup logic can be
designed around what's actually causing the mismatches.

Note: TMDB rows in combined_movies.csv don't carry release_date (it was
dropped when TMDB was standardized onto the MovieLens schema), so this
script pulls it back from data/raw/tmdb_movies.csv, joined on title.
"""

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

pd.set_option("display.max_colwidth", 45)
pd.set_option("display.width", 200)


def strip_year_suffix(title):
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


def normalize_basic(title):
    return re.sub(r"\s+", " ", title).strip().lower()


def strip_punct(title):
    return re.sub(r"[^a-z0-9\s]", "", title.lower())


def swap_leading_article(title):
    m = re.match(r"^(The|A|An)\s+(.+)$", title, flags=re.IGNORECASE)
    if m:
        return f"{m.group(2)}, {m.group(1)}"
    return title


def classify_pair(ml_title_raw, tmdb_title_raw):
    ml_clean = strip_year_suffix(ml_title_raw)
    tmdb_clean = tmdb_title_raw.strip()

    if ml_clean == tmdb_clean:
        return "exact"
    if ml_clean.lower() == tmdb_clean.lower():
        return "case_only"
    if strip_punct(ml_clean) == strip_punct(tmdb_clean):
        return "punctuation_or_special_chars"
    if normalize_basic(swap_leading_article(ml_clean)) == normalize_basic(tmdb_clean) or normalize_basic(
        ml_clean
    ) == normalize_basic(swap_leading_article(tmdb_clean)):
        return "article_position"
    return "other_fuzzy"


def main():
    combined_movies = pd.read_csv(RAW_DIR / "combined_movies.csv")
    tmdb_movies = pd.read_csv(RAW_DIR / "tmdb_movies.csv")[["title", "release_date"]]
    tmdb_movies = tmdb_movies.drop_duplicates(subset="title")

    ml_flagged = combined_movies[
        (combined_movies["source"] == "movielens") & (combined_movies["is_duplicate_title_year"])
    ].copy()
    tmdb_flagged = combined_movies[
        (combined_movies["source"] == "tmdb") & (combined_movies["is_duplicate_title_year"])
    ].copy()
    print(f"Flagged MovieLens rows in combined_movies.csv: {len(ml_flagged):,}")
    print(f"Flagged TMDB rows in combined_movies.csv:      {len(tmdb_flagged):,}")

    ml_flagged["year"] = ml_flagged["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("Int64")
    ml_flagged["title_norm"] = ml_flagged["title"].apply(strip_year_suffix).str.lower()

    tmdb_flagged = tmdb_flagged.merge(tmdb_movies, on="title", how="left")
    tmdb_flagged["year"] = pd.to_datetime(tmdb_flagged["release_date"], errors="coerce").dt.year.astype("Int64")
    tmdb_flagged["title_norm"] = tmdb_flagged["title"].str.strip().str.lower()

    pairs = ml_flagged.merge(tmdb_flagged, on=["title_norm", "year"], suffixes=("_ml", "_tmdb"), how="inner")
    keys = pairs.drop_duplicates(subset=["title_norm", "year"]).copy()
    print(f"\nDistinct (title, year) duplicate keys recovered here: {len(keys):,}  (859 reported by merge_datasets.py)")
    if len(pairs) > len(keys):
        print(f"({len(pairs):,} total row-pairs -- some keys match more than one row on a side)")

    keys["match_type"] = keys.apply(lambda r: classify_pair(r["title_ml"], r["title_tmdb"]), axis=1)

    print("\n" + "=" * 70)
    print("1. SAMPLE OF 20 FLAGGED PAIRS -- MovieLens vs TMDB, side by side")
    print("=" * 70)
    sample = keys.sample(n=min(20, len(keys)), random_state=42).sort_values("match_type")
    print(
        sample[["title_ml", "title_tmdb", "year", "match_type", "movieId_ml", "movieId_tmdb"]].to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("2/3. MISMATCH TYPE BREAKDOWN (by distinct duplicate key)")
    print("=" * 70)
    counts = keys["match_type"].value_counts()
    print(counts.to_string())
    n_exact = int(counts.get("exact", 0))
    n_fuzzy = len(keys) - n_exact
    print(f"\nExact title matches: {n_exact:,} ({n_exact / len(keys) * 100:.1f}%)")
    print(f"Fuzzy title matches: {n_fuzzy:,} ({n_fuzzy / len(keys) * 100:.1f}%)")

    for cat in ["case_only", "punctuation_or_special_chars", "article_position", "other_fuzzy"]:
        subset = keys[keys["match_type"] == cat]
        if len(subset) == 0:
            continue
        print(f"\n-- {cat} ({len(subset)} keys) -- examples --")
        print(subset[["title_ml", "title_tmdb"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
