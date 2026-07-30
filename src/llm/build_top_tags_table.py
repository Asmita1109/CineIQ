"""Precomputes a compact top-3-genome-tags-per-movie table from
genome_scores_clean.csv + genome_tags_clean.csv.

RecommendationExplainer previously loaded the entire 18.47M-row
genome_scores_clean.csv into memory at runtime just to look up ~3 rows per
movie per request -- that alone cost an estimated 500MB-1GB+ RSS and was
the leading suspect in a Streamlit Cloud OOM crash (confirmed by an HTTP
503 from the platform, not a catchable Python exception). This script
runs once, offline, and its ~16K-row output is what explainer.py loads
instead -- a >99% reduction in rows for the same information.

Run with: python src/llm/build_top_tags_table.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TOP_N_TAGS = 3


def main():
    print("Loading genome_scores_clean.csv (18.47M rows) ...")
    genome_scores = pd.read_csv(PROCESSED_DIR / "genome_scores_clean.csv")
    print(f"  {len(genome_scores):,} rows, {genome_scores['movieId'].nunique():,} movies")

    print("Loading genome_tags_clean.csv ...")
    genome_tags = pd.read_csv(PROCESSED_DIR / "genome_tags_clean.csv")
    tag_names = genome_tags.set_index("tagId")["tag"].to_dict()

    print(f"Computing top {TOP_N_TAGS} tags per movie ...")
    top = genome_scores.sort_values("relevance", ascending=False).groupby("movieId").head(TOP_N_TAGS)

    rows = []
    for movie_id, group in top.groupby("movieId"):
        tags = [tag_names.get(int(tag_id), f"tag {tag_id}") for tag_id in group["tagId"]]
        tags += [""] * (TOP_N_TAGS - len(tags))  # pad in the rare case a movie has < 3 tag rows
        rows.append({"movieId": movie_id, "tag_1": tags[0], "tag_2": tags[1], "tag_3": tags[2]})

    result = pd.DataFrame(rows)
    output_path = PROCESSED_DIR / "movie_top_tags.csv"
    result.to_csv(output_path, index=False)
    print(f"Saved -> {output_path} ({len(result):,} rows, {output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
