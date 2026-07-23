"""Clean and preprocess the raw MovieLens 'ml-latest' dataset."""

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "ml-latest"
PROCESSED_DIR = DATA_DIR / "processed"

MIN_RATINGS_PER_USER = 5
MIN_RATINGS_PER_MOVIE = 5
VALID_RATING_MIN = 0.5
VALID_RATING_MAX = 5.0

# "Matrix, The (1999)" -> "The Matrix (1999)"
ARTICLE_SUFFIX_RE = re.compile(r"^(.*),\s+(The|A|An)\s+\((\d{4})\)\s*$")


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
    links = pd.read_csv(RAW_DIR / "links.csv")
    genome_scores = pd.read_csv(
        RAW_DIR / "genome-scores.csv",
        dtype={"movieId": "int32", "tagId": "int16", "relevance": "float32"},
    )
    genome_tags = pd.read_csv(RAW_DIR / "genome-tags.csv", dtype={"tagId": "int16"})
    return ratings, movies, tags, links, genome_scores, genome_tags


def report_missing_and_duplicates(df, name):
    print(f"\n{name}: {len(df):,} rows")
    missing = df.isna().sum()
    total_missing = int(missing.sum())
    print("  Missing values:")
    if total_missing == 0:
        print("    none")
    else:
        for col, count in missing[missing > 0].items():
            print(f"    {col}: {count:,}")
    dup = int(df.duplicated().sum())
    print(f"  Duplicate rows: {dup:,}")
    return dup


def report_quality(ratings, movies, tags, links, genome_scores, genome_tags):
    print("=" * 70)
    print("DATA QUALITY REPORT (before cleaning)")
    print("=" * 70)

    report_missing_and_duplicates(ratings, "ratings.csv")
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

    report_missing_and_duplicates(movies, "movies.csv")
    n_needs_retitle = int(movies["title"].dropna().apply(lambda t: bool(ARTICLE_SUFFIX_RE.match(t.strip()))).sum())
    print(f"  Titles in 'Article, The (year)' form needing reformat: {n_needs_retitle:,}")

    report_missing_and_duplicates(tags, "tags.csv")

    report_missing_and_duplicates(links, "links.csv")
    missing_tmdb = int(links["tmdbId"].isna().sum())
    print(
        f"  tmdbId missing: {missing_tmdb:,} ({missing_tmdb / len(links) * 100:.2f}%) "
        f"-- will be flagged, not dropped"
    )

    report_missing_and_duplicates(genome_scores, "genome-scores.csv")
    report_missing_and_duplicates(genome_tags, "genome-tags.csv")


def fix_title_article(title):
    if not isinstance(title, str):
        return title
    m = ARTICLE_SUFFIX_RE.match(title.strip())
    if not m:
        return title
    base, article, year = m.groups()
    return f"{article} {base.strip()} ({year})"


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

    n_retitled = int(
        movies["title"].dropna().apply(lambda t: bool(ARTICLE_SUFFIX_RE.match(t.strip()))).sum()
    )
    movies["title"] = movies["title"].apply(fix_title_article)
    stats["titles_reformatted"] = n_retitled

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


def clean_links(links):
    stats = {"start": len(links)}

    links = links.drop_duplicates().copy()
    stats["after_dedup"] = len(links)

    links["movieId"] = links["movieId"].astype("int32")
    links["imdbId"] = links["imdbId"].astype("int32")
    links["tmdb_id_missing"] = links["tmdbId"].isna()
    stats["tmdb_missing_flagged"] = int(links["tmdb_id_missing"].sum())
    links["tmdbId"] = links["tmdbId"].astype("Int64")  # nullable int -- keeps the NaNs, drops nothing

    return links, stats


def clean_genome_scores(genome_scores):
    stats = {"start": len(genome_scores)}
    genome_scores = genome_scores.drop_duplicates().copy()
    stats["after_dedup"] = len(genome_scores)
    return genome_scores, stats


def clean_genome_tags(genome_tags):
    stats = {"start": len(genome_tags)}
    genome_tags = genome_tags.drop_duplicates().copy()
    stats["after_dedup"] = len(genome_tags)
    return genome_tags, stats


def print_before_after_report(ratings_stats, movies_stats, tags_stats, links_stats, genome_scores_stats, genome_tags_stats):
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
    print(f"  - titles reformatted ('Article, The (yr)' -> 'The Article (yr)'): {m['titles_reformatted']:,}")
    print(f"  After:  {m['after_dedup']:,} rows")

    t = tags_stats
    print("\ntags.csv")
    print(f"  Before: {t['start']:,} rows")
    print(f"  - duplicates dropped: {t['start'] - t['after_dedup']:,}")
    print(f"  After:  {t['after_dedup']:,} rows")

    l = links_stats
    print("\nlinks.csv")
    print(f"  Before: {l['start']:,} rows")
    print(f"  - duplicates dropped: {l['start'] - l['after_dedup']:,}")
    print(f"  - rows flagged with missing tmdbId (kept, not dropped): {l['tmdb_missing_flagged']:,}")
    print(f"  After:  {l['after_dedup']:,} rows")

    gs = genome_scores_stats
    print("\ngenome-scores.csv")
    print(f"  Before: {gs['start']:,} rows")
    print(f"  - duplicates dropped: {gs['start'] - gs['after_dedup']:,}")
    print(f"  After:  {gs['after_dedup']:,} rows")

    gt = genome_tags_stats
    print("\ngenome-tags.csv")
    print(f"  Before: {gt['start']:,} rows")
    print(f"  - duplicates dropped: {gt['start'] - gt['after_dedup']:,}")
    print(f"  After:  {gt['after_dedup']:,} rows")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data from", RAW_DIR)
    ratings, movies, tags, links, genome_scores, genome_tags = load_raw()
    print(f"  ratings.csv:        {ratings.shape}")
    print(f"  movies.csv:         {movies.shape}")
    print(f"  tags.csv:           {tags.shape}")
    print(f"  links.csv:          {links.shape}")
    print(f"  genome-scores.csv:  {genome_scores.shape}")
    print(f"  genome-tags.csv:    {genome_tags.shape}")

    report_quality(ratings, movies, tags, links, genome_scores, genome_tags)

    print("\n" + "=" * 70)
    print("CLEANING")
    print("=" * 70)

    ratings_clean, ratings_stats = clean_ratings(ratings)
    movies_clean, movies_stats = clean_movies(movies)
    tags_clean, tags_stats = clean_tags(tags)
    links_clean, links_stats = clean_links(links)
    genome_scores_clean, genome_scores_stats = clean_genome_scores(genome_scores)
    genome_tags_clean, genome_tags_stats = clean_genome_tags(genome_tags)

    print("\nSaving cleaned files to", PROCESSED_DIR)

    ratings_clean.drop(columns=["date"]).to_csv(PROCESSED_DIR / "ratings_clean.csv", index=False)
    print(f"  Wrote ratings_clean.csv ({len(ratings_clean):,} rows)")

    movies_clean.to_csv(PROCESSED_DIR / "movies_clean.csv", index=False)
    print(f"  Wrote movies_clean.csv ({len(movies_clean):,} rows)")

    tags_clean.drop(columns=["date"]).to_csv(PROCESSED_DIR / "tags_clean.csv", index=False)
    print(f"  Wrote tags_clean.csv ({len(tags_clean):,} rows)")

    links_clean.to_csv(PROCESSED_DIR / "links_clean.csv", index=False)
    print(f"  Wrote links_clean.csv ({len(links_clean):,} rows)")

    genome_scores_clean.to_csv(PROCESSED_DIR / "genome_scores_clean.csv", index=False)
    print(f"  Wrote genome_scores_clean.csv ({len(genome_scores_clean):,} rows)")

    genome_tags_clean.to_csv(PROCESSED_DIR / "genome_tags_clean.csv", index=False)
    print(f"  Wrote genome_tags_clean.csv ({len(genome_tags_clean):,} rows)")

    print_before_after_report(
        ratings_stats, movies_stats, tags_stats, links_stats, genome_scores_stats, genome_tags_stats
    )


if __name__ == "__main__":
    main()
