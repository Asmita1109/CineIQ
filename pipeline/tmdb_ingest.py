"""Pull recent/upcoming movie metadata from TMDB and format it MovieLens-style.

Fetches every movie TMDB's discover/movie endpoint returns for release dates
between START_DATE and END_DATE, maps genre IDs to names, and writes two CSVs
shaped like the MovieLens files the rest of the pipeline already expects:
tmdb_movies.csv (movieId/title/genres, plus TMDB's own metadata) and
tmdb_ratings.csv (synthetic userId/movieId/rating/timestamp rows standing in
for real user ratings, derived from each movie's vote_average/vote_count).
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ENV_PATH = PROJECT_ROOT / ".env"

TMDB_BASE_URL = "https://api.themoviedb.org/3"
START_DATE = "2019-01-01"
END_DATE = "2026-12-31"
MAX_PAGES_PER_QUERY = 500  # TMDB caps discover/movie at 500 pages (10,000 results)
REQUEST_DELAY_SECONDS = 0.05
SYNTHETIC_USER_POOL = 50_000
RATING_SCALE_MIN = 0.5
RATING_SCALE_MAX = 5.0
RANDOM_STATE = 42


def load_env_file():
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def get_api_key():
    load_env_file()
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TMDB_API_KEY not set. Add it to a .env file at the project root "
            "(TMDB_API_KEY=...) or export it as an environment variable."
        )
    return api_key


_session = None


def get_session():
    global _session
    if _session is None:
        session = requests.Session()
        retry = Retry(
            total=8,
            connect=8,
            read=8,
            status=8,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        session.mount("https://", adapter)
        _session = session
    return _session


def tmdb_get(path, api_key, params=None, max_attempts=3):
    params = dict(params or {})
    params["api_key"] = api_key
    url = f"{TMDB_BASE_URL}{path}"

    last_exc = None
    for attempt in range(max_attempts):
        try:
            response = get_session().get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            time.sleep(1 + attempt)

    raise RuntimeError(f"Failed to fetch {url} after {max_attempts} attempts") from last_exc


def fetch_genre_map(api_key):
    data = tmdb_get("/genre/movie/list", api_key, params={"language": "en-US"})
    return {g["id"]: g["name"] for g in data["genres"]}


def discover_page(api_key, gte, lte, page):
    return tmdb_get(
        "/discover/movie",
        api_key,
        params={
            "primary_release_date.gte": gte,
            "primary_release_date.lte": lte,
            "sort_by": "popularity.desc",
            "include_adult": "false",
            "page": page,
        },
    )


def split_date_range(gte, lte):
    start, end = pd.Timestamp(gte), pd.Timestamp(lte)
    mid = start + pd.Timedelta(days=(end - start).days // 2)
    return (gte, mid.strftime("%Y-%m-%d")), ((mid + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), lte)


def fetch_movies_for_range(api_key, gte, lte, depth=0):
    """Fetch every movie in [gte, lte]. TMDB caps discover/movie at 500 pages
    (10,000 results) per query, so any range that exceeds the cap is recursively
    split in half by date until each leaf query fits under it."""
    data = discover_page(api_key, gte, lte, page=1)
    total_pages = data.get("total_pages", 1)
    total_results = data.get("total_results", 0)
    span_days = (pd.Timestamp(lte) - pd.Timestamp(gte)).days

    if total_pages > MAX_PAGES_PER_QUERY and span_days > 0:
        (g1, l1), (g2, l2) = split_date_range(gte, lte)
        print(
            f"{'  ' * depth}{gte}..{lte}: {total_results:,} results ({total_pages} pages) "
            f"> cap, splitting into {g1}..{l1} / {g2}..{l2}"
        )
        return fetch_movies_for_range(api_key, g1, l1, depth + 1) + fetch_movies_for_range(
            api_key, g2, l2, depth + 1
        )

    results = list(data.get("results", []))
    for page in range(2, min(total_pages, MAX_PAGES_PER_QUERY) + 1):
        page_data = discover_page(api_key, gte, lte, page=page)
        results.extend(page_data.get("results", []))
        time.sleep(REQUEST_DELAY_SECONDS)

    if total_pages > MAX_PAGES_PER_QUERY:
        print(
            f"{'  ' * depth}Warning: {gte}..{lte} still has {total_results:,} results "
            f"on a single day; only the first {MAX_PAGES_PER_QUERY} pages were pulled."
        )

    print(f"{'  ' * depth}{gte}..{lte}: {len(results):,} movies fetched ({total_results:,} reported)")
    return results


def fetch_all_movies(api_key):
    return fetch_movies_for_range(api_key, START_DATE, END_DATE)


def build_movies_dataframe(raw_movies, genre_map):
    rows = []
    seen_ids = set()
    for m in raw_movies:
        tmdb_id = m.get("id")
        if tmdb_id is None or tmdb_id in seen_ids:
            continue
        seen_ids.add(tmdb_id)

        genre_names = [genre_map.get(gid, "Unknown") for gid in m.get("genre_ids", [])]
        genres = "|".join(genre_names) if genre_names else "(no genres listed)"

        rows.append(
            {
                "movieId": tmdb_id,
                "title": m.get("title", ""),
                "genres": genres,
                "release_date": m.get("release_date", ""),
                "popularity": m.get("popularity", 0.0),
                "vote_average": m.get("vote_average", 0.0),
                "vote_count": m.get("vote_count", 0),
            }
        )
    return pd.DataFrame(rows)


def build_ratings_dataframe(movies_df, random_state=RANDOM_STATE):
    """One synthetic rating row per TMDB vote: rating = vote_average (rescaled to
    the MovieLens 0.5-5.0 scale), timestamp = release_date, userId = random draw
    from a synthetic user pool."""
    rng = np.random.default_rng(random_state)
    frames = []

    for row in movies_df.itertuples(index=False):
        vote_count = int(row.vote_count) if pd.notna(row.vote_count) else 0
        if vote_count <= 0:
            continue
        try:
            timestamp = int(pd.Timestamp(row.release_date).timestamp())
        except (ValueError, TypeError):
            continue

        rating = round(row.vote_average) / 2  # 0-10 -> nearest 0.5 on a 0.5-5 scale
        rating = min(max(rating, RATING_SCALE_MIN), RATING_SCALE_MAX)

        frames.append(
            pd.DataFrame(
                {
                    "userId": rng.integers(1, SYNTHETIC_USER_POOL + 1, size=vote_count),
                    "movieId": row.movieId,
                    "rating": rating,
                    "timestamp": timestamp,
                }
            )
        )

    if not frames:
        return pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
    return pd.concat(frames, ignore_index=True)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api_key = get_api_key()

    print("Fetching TMDB genre list...")
    genre_map = fetch_genre_map(api_key)
    print(f"  {len(genre_map)} genres loaded")

    print(f"\nFetching movies released {START_DATE} to {END_DATE} (recursive date-range pagination)...")
    raw_movies = fetch_all_movies(api_key)
    print(f"\nTotal raw results collected: {len(raw_movies):,}")

    movies_df = build_movies_dataframe(raw_movies, genre_map)
    print(f"Unique movies after de-duplication: {len(movies_df):,}")

    print("\nGenerating synthetic ratings from vote_average / vote_count...")
    ratings_df = build_ratings_dataframe(movies_df)
    print(f"Synthetic ratings generated: {len(ratings_df):,}")

    movies_path = RAW_DIR / "tmdb_movies.csv"
    ratings_path = RAW_DIR / "tmdb_ratings.csv"
    movies_df.to_csv(movies_path, index=False)
    ratings_df.to_csv(ratings_path, index=False)

    print(f"\nSaved -> {movies_path}")
    print(f"tmdb_movies.csv: shape = {movies_df.shape}")
    print(movies_df.head())

    print(f"\nSaved -> {ratings_path}")
    print(f"tmdb_ratings.csv: shape = {ratings_df.shape}")
    print(ratings_df.head())


if __name__ == "__main__":
    main()
