"""Generates short, personalized natural-language explanations for
recommendations via the Claude API.

This module doesn't compute recommendations itself -- it takes a
(user_id, movie_id) pair plus the NCF relevance score that already produced
that recommendation upstream (src/recommender), builds a structured prompt
from that user's profile, the movie's genre/tag profile, and whether the
movie's genre is currently trending, and asks Claude for a 2-3 sentence
explanation to show next to the recommendation.

Two data sources are used beyond the four named in this module's spec,
because neither of those four alone can produce what the prompt needs:
  - movies_clean.csv, for movie title (movie_features.parquet has genres
    and rating stats but no title column).
  - genome_scores_clean.csv, for each movie's raw per-tag relevance scores
    (genome_tags_clean.csv only maps tagId -> tag name; movie_features.parquet
    only has the 50-dim SVD-*compressed* genome embedding, which has no
    per-tag meaning to rank "top tags" from).
"""

import os
from pathlib import Path

import anthropic
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
PROCESSED_DIR = DATA_DIR / "processed"
ENV_PATH = PROJECT_ROOT / ".env"

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 250
TOP_N_TAGS = 3

RECENT_MOVIE_COLS = [f"recent_movie_{i}" for i in range(1, 6)]


def load_env_file():
    """Same lightweight .env loader as pipeline/tmdb_ingest.py -- no new
    dependency, and os.environ.setdefault means an already-exported
    ANTHROPIC_API_KEY always wins over the .env file."""
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
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to a .env file at the project root "
            "(ANTHROPIC_API_KEY=...) or export it as an environment variable."
        )
    return api_key


class RecommendationExplainer:
    """Loads user/movie/genome/forecasting reference data once and reuses it
    across many explain()/batch_explain() calls -- those calls just look up
    already-indexed data and hit the Claude API, no per-call file re-reads.
    """

    def __init__(
        self,
        user_features_path=FEATURES_DIR / "user_features.parquet",
        movie_features_path=FEATURES_DIR / "movie_features.parquet",
        movies_path=PROCESSED_DIR / "movies_clean.csv",
        genome_scores_path=PROCESSED_DIR / "genome_scores_clean.csv",
        genome_tags_path=PROCESSED_DIR / "genome_tags_clean.csv",
        forecasting_features_path=FEATURES_DIR / "forecasting_features.csv",
        model=MODEL,
        api_key=None,
    ):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or get_api_key())

        user_features = pd.read_parquet(
            user_features_path,
            columns=["userId", "user_segment", "favorite_genre", "avg_rating", *RECENT_MOVIE_COLS],
        )
        self.user_features = user_features.set_index("userId").to_dict("index")

        movie_features = pd.read_parquet(movie_features_path, columns=["movieId", "genres"])
        movies = pd.read_csv(movies_path, usecols=["movieId", "title"])
        movie_info = movie_features.merge(movies, on="movieId", how="left")
        self.movie_info = movie_info.set_index("movieId").to_dict("index")

        genome_tags = pd.read_csv(genome_tags_path)
        self.tag_names = genome_tags.set_index("tagId")["tag"].to_dict()

        # Raw per-(movie, tag) relevance -- kept indexed by movieId so a
        # single movie's ~1,128 tag rows can be pulled out without scanning
        # all 18M+ rows; top tags are then computed lazily per movie
        # (memoized in _top_tags_cache) rather than precomputed for every
        # movie in the catalog up front.
        self.genome_scores = pd.read_csv(genome_scores_path).set_index("movieId")
        self._top_tags_cache = {}

        self.trending_genre, self.trending_genre_score = self._latest_trending_genre(forecasting_features_path)

        self._cache = {}  # (user_id, movie_id) -> explanation string

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _latest_trending_genre(forecasting_features_path):
        """The genre with the most ratings in the most recent week on
        record -- "currently trending" for the purposes of an explanation
        written today."""
        df = pd.read_csv(forecasting_features_path, parse_dates=["week_start"])
        latest = df[df["week_start"] == df["week_start"].max()]
        top_row = latest.loc[latest["rating_count"].idxmax()]
        return str(top_row["genre"]), float(top_row["rating_count"])

    def _top_tags(self, movie_id):
        if movie_id in self._top_tags_cache:
            return self._top_tags_cache[movie_id]
        if movie_id not in self.genome_scores.index:
            tags = []
        else:
            rows = self.genome_scores.loc[[movie_id]].nlargest(TOP_N_TAGS, "relevance")
            tags = [self.tag_names.get(int(tag_id), f"tag {tag_id}") for tag_id in rows["tagId"]]
        self._top_tags_cache[movie_id] = tags
        return tags

    # ------------------------------------------------------------------
    # Profile builders
    # ------------------------------------------------------------------
    def _user_profile(self, user_id):
        info = self.user_features.get(user_id)
        if info is None:
            return {"segment": "regular", "favorite_genre": "unknown", "avg_rating": None, "recent_movie_titles": []}
        recent_ids = [info[c] for c in RECENT_MOVIE_COLS if info[c] is not None and not pd.isna(info[c])]
        recent_titles = [self.movie_info.get(int(m), {}).get("title", f"movie {m}") for m in recent_ids]
        return {
            "segment": info["user_segment"],
            "favorite_genre": info["favorite_genre"],
            "avg_rating": float(info["avg_rating"]),
            "recent_movie_titles": recent_titles,
        }

    def _movie_profile(self, movie_id):
        info = self.movie_info.get(movie_id, {})
        return {
            "title": info.get("title") or f"movie {movie_id}",
            "genres": info.get("genres") or "",
            "top_tags": self._top_tags(movie_id),
        }

    def _trending_signal(self, movie_genres):
        genre_list = (movie_genres or "").split("|")
        return {
            "trending_genre": self.trending_genre,
            "trending_genre_score": self.trending_genre_score,
            "movie_matches_trending_genre": self.trending_genre in genre_list,
        }

    # ------------------------------------------------------------------
    # Prompt + explanation
    # ------------------------------------------------------------------
    def build_prompt(self, user_id, movie_id, ncf_score):
        user = self._user_profile(user_id)
        movie = self._movie_profile(movie_id)
        trend = self._trending_signal(movie["genres"])

        recent = ", ".join(user["recent_movie_titles"]) if user["recent_movie_titles"] else "no recent ratings on file"
        tags = ", ".join(movie["top_tags"]) if movie["top_tags"] else "no distinctive tags on file"
        avg_rating = f"{user['avg_rating']:.1f}/5" if user["avg_rating"] is not None else "unknown"
        trend_note = f"{trend['trending_genre']} is the fastest-growing genre this week"
        if trend["movie_matches_trending_genre"]:
            trend_note += f", and this movie is a {trend['trending_genre']} title"

        return f"""You are writing a short, friendly explanation for why a movie was recommended to a user on a streaming platform.

USER PROFILE
- Segment: {user['segment']}
- Favorite genre: {user['favorite_genre']}
- Average rating given: {avg_rating}
- Recently rated: {recent}

RECOMMENDED MOVIE
- Title: {movie['title']}
- Genres: {movie['genres']}
- Top tags: {tags}

WHY IT WAS RECOMMENDED
- Model relevance score: {ncf_score:.3f} (higher = more relevant to this user, from a collaborative filtering model)
- Genre trend: {trend_note}

Write a 2-3 sentence explanation, addressed to the user, for why this movie was recommended. Be specific and reference their taste profile and/or the trend signal where relevant. Do not mention model names, scores, or technical details -- write as a friendly recommendation, not a technical report.

Write only the explanation sentences. No headers, no preamble, no labels like "Here is" or "Why we think". Start directly with the explanation."""

    def explain(self, user_id, movie_id, ncf_score):
        """Returns a 2-3 sentence explanation for recommending movie_id to
        user_id, given that recommendation's NCF relevance score. Caches by
        (user_id, movie_id) so the same pair never calls the API twice in
        this instance's lifetime."""
        cache_key = (user_id, movie_id)
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = self.build_prompt(user_id, movie_id, ncf_score)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        explanation = response.content[0].text.strip()
        self._cache[cache_key] = explanation
        return explanation

    def batch_explain(self, recommendations):
        """recommendations: list of dicts, each with 'user_id', 'movie_id',
        and 'ncf_score' keys. Returns a list of the same dicts with an
        'explanation' key added, in the same order."""
        results = []
        for rec in recommendations:
            explanation = self.explain(rec["user_id"], rec["movie_id"], rec["ncf_score"])
            results.append({**rec, "explanation": explanation})
        return results
