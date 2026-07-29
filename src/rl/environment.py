"""Simulated recommendation environment for CineIQ's contextual bandit RL
layer.

Each "round" is one historical (userId, timestamp) interaction from
rl_train.parquet or rl_test.parquet, walked through in temporal order. For
that round the environment exposes:
  - a raw state (user segment, last 5 rated movies, current trending genre)
  - an action space (top 10 movies scored by the trained BPR NCF recommender
    for that user -- the candidates a real recommendation slot would show)

Reward for choosing any of those candidates is looked up from historical
data (rl_features.parquet, the fullest interaction log in the project):
reward = 1 if that user actually rated that movie >= 4.0 at some point,
else 0 (covers both "rated it low" and "never rated it"). This lets the
bandit be scored on candidates it picks that differ from whatever movieId
the current split row logged -- the row only anchors *when* the round
happens (userId, timestamp), not which action is being evaluated.

Building a raw state (segment/history/genre) is this module's job; turning
that into a numeric context vector for a specific agent is that agent's job
(see agent.LinUCBAgent.build_context) -- keeps "what the state is" separate
from "how a given agent encodes it".
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"

# environment.py lives in src/rl/, but scoring the action space needs the
# trained BPR NCF model defined in src/recommender/ -- add that directory to
# sys.path so we can import it directly, same bare-import convention the
# rest of the codebase uses within a single package directory.
RECOMMENDER_SRC = PROJECT_ROOT / "src" / "recommender"
sys.path.insert(0, str(RECOMMENDER_SRC))
from model import NCF  # noqa: E402
from evaluate import build_genome_lookup  # noqa: E402

DEFAULT_BPR_MODEL_PATH = MODELS_DIR / "recommender_model_bpr.pt"
DEFAULT_RATED_HISTORY_PATH = FEATURES_DIR / "rl_features.parquet"

SEGMENTS = ["casual", "regular", "power"]

# Fixed, sorted vocabulary of genres tracked by the forecasting model --
# keeps the trending-genre one-hot encoding (built in agent.py) a stable
# size/order regardless of which genres happen to be trending in any given
# run. Matches the genre set in forecasting_features.csv.
GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

ACTION_SPACE_SIZE = 10
RECENT_MOVIE_COLS = [f"recent_movie_{i}" for i in range(1, 6)]


class RecommendationEnv:
    """Steps through historical interactions in temporal order, exposing a
    state/action-space/reward interface for a contextual bandit.

    Usage:
        env = RecommendationEnv(interactions_path=FEATURES_DIR / "rl_train.parquet")
        for user_id, timestamp in env.iter_interactions():
            state = env.get_state(user_id, timestamp)
            candidates = env.get_action_space(user_id)
            action = agent.select_action(agent.build_context(state), candidates)
            reward = env.step(user_id, action)
    """

    def __init__(
        self,
        interactions_path,
        user_features_path=FEATURES_DIR / "user_features.parquet",
        movie_features_path=FEATURES_DIR / "movie_features.parquet",
        forecasting_features_path=FEATURES_DIR / "forecasting_features.csv",
        rated_history_path=DEFAULT_RATED_HISTORY_PATH,
        bpr_model_path=DEFAULT_BPR_MODEL_PATH,
        top_k=ACTION_SPACE_SIZE,
        max_interactions=None,
        device=None,
    ):
        self.top_k = top_k
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading interactions: {interactions_path}")
        data = pd.read_parquet(interactions_path, columns=["userId", "movieId", "timestamp"])
        data = data.sort_values("timestamp").reset_index(drop=True)
        if max_interactions is not None:
            data = data.iloc[:max_interactions]
        self.data = data

        print(f"Loading user_features: {user_features_path}")
        user_features = pd.read_parquet(
            user_features_path, columns=["userId", "user_segment", *RECENT_MOVIE_COLS]
        )
        self.user_state = user_features.set_index("userId").to_dict("index")

        print(f"Loading movie_features: {movie_features_path}")
        self.movie_features = pd.read_parquet(movie_features_path)
        self.catalog_movie_ids = self.movie_features["movieId"].to_numpy()
        self._most_popular_movie_id = int(
            self.movie_features.loc[self.movie_features["total_ratings"].idxmax(), "movieId"]
        )

        print(f"Loading forecasting_features: {forecasting_features_path}")
        self._week_starts, self._trend_genres, self._trend_scores = self._build_weekly_trend_table(
            forecasting_features_path
        )

        print(f"Loading reward history: {rated_history_path}")
        history = pd.read_parquet(rated_history_path, columns=["userId", "movieId", "reward"])
        self._reward_keys, self._reward_values, self._movie_id_bound = self._build_reward_lookup(
            history, self.catalog_movie_ids
        )

        print(f"Loading BPR NCF checkpoint: {bpr_model_path}")
        self.model, self.user_id_map, self.movie_id_map, self.genome_lookup = self._load_bpr_model(
            bpr_model_path, self.movie_features
        )
        self.idx_to_movie_id = {idx: mid for mid, idx in self.movie_id_map.items()}
        self.num_movies = len(self.movie_id_map)
        self._action_cache = {}

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_weekly_trend_table(forecasting_features_path):
        """For each week, the genre with the most ratings that week (the
        "current trending genre"). Returns three aligned numpy arrays
        (sorted by week) for fast nearest-week-<=-timestamp lookup via
        np.searchsorted."""
        df = pd.read_csv(forecasting_features_path, parse_dates=["week_start"])
        top_idx = df.groupby("week_start")["rating_count"].idxmax()
        top = df.loc[top_idx, ["week_start", "genre", "rating_count"]].sort_values("week_start")
        week_starts = top["week_start"].to_numpy(dtype="datetime64[s]").astype("int64")
        return week_starts, top["genre"].to_numpy(), top["rating_count"].to_numpy(dtype="float64")

    @staticmethod
    def _build_reward_lookup(history, catalog_movie_ids):
        """(userId, movieId) -> reward, without a python dict of 33M+ tuple
        keys (measured ~7GB RSS -- the actual cause of this environment
        getting OOM-killed on full rl_train.parquet runs). Instead, encode
        each pair as a single sortable int64 key (userId * bound + movieId,
        bound set from the full movie catalog so any queried movieId is
        guaranteed to fit) and binary-search a sorted array of those keys --
        well under 1GB for the same 33.7M rows."""
        movie_id_bound = int(catalog_movie_ids.max()) + 1
        keys = (
            history["userId"].to_numpy(dtype="int64") * movie_id_bound
            + history["movieId"].to_numpy(dtype="int64")
        )
        order = np.argsort(keys, kind="stable")
        return keys[order], history["reward"].to_numpy()[order], movie_id_bound

    @staticmethod
    def _load_bpr_model(bpr_model_path, movie_features):
        ckpt = torch.load(bpr_model_path, map_location="cpu", weights_only=False)
        model = NCF(
            num_users=ckpt["num_users"],
            num_movies=ckpt["num_movies"],
            embedding_dim=ckpt["embedding_dim"],
            genome_dim=ckpt["genome_dim"],
            hidden_layers=ckpt["hidden_layers"],
            dropout=ckpt.get("dropout", 0.2),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        genome_lookup = build_genome_lookup(movie_features, ckpt["movie_id_map"], ckpt["genome_dim"])
        return model, ckpt["user_id_map"], ckpt["movie_id_map"], genome_lookup

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------
    def iter_interactions(self, start_index=0):
        """Yields (userId, timestamp) for every interaction in temporal
        order -- the sequence of rounds a training/eval loop steps through.
        The row's own movieId isn't yielded: the agent/baseline each pick
        their own action every round, and step() scores whatever action was
        actually chosen against historical data.

        start_index skips the first N rows so a training loop can resume
        after a checkpoint without redoing already-processed interactions."""
        for row in self.data.iloc[start_index:].itertuples(index=False):
            yield row.userId, row.timestamp

    def get_state(self, user_id, timestamp):
        info = self.user_state.get(user_id)
        segment = info["user_segment"] if info is not None else "regular"
        recent_movies = (
            [info[c] for c in RECENT_MOVIE_COLS] if info is not None else [None] * len(RECENT_MOVIE_COLS)
        )
        trending_genre, trending_genre_score = self._trending_genre_for_timestamp(timestamp)
        return {
            "user_id": user_id,
            "user_segment": segment,
            "recent_movies": recent_movies,
            "trending_genre": trending_genre,
            "trending_genre_score": trending_genre_score,
        }

    def _trending_genre_for_timestamp(self, timestamp):
        idx = np.searchsorted(self._week_starts, timestamp, side="right") - 1
        idx = int(np.clip(idx, 0, len(self._week_starts) - 1))
        return str(self._trend_genres[idx]), float(self._trend_scores[idx])

    @torch.no_grad()
    def _score_all_movies_for_user(self, user_idx):
        n = self.num_movies
        user_idx_t = torch.full((n,), user_idx, dtype=torch.long, device=self.device)
        movie_idx_t = torch.arange(n, dtype=torch.long, device=self.device)
        genome_t = torch.from_numpy(self.genome_lookup).to(self.device)
        scores = self.model(user_idx_t, movie_idx_t, genome_t).cpu().numpy()
        return scores

    def get_action_space(self, user_id):
        """Top `top_k` movieIds by BPR relevance score for this user,
        memoized per user (the model is frozen, so this is identical across
        every round for the same user -- scoring the full catalog per round
        would be wasteful)."""
        if user_id in self._action_cache:
            return self._action_cache[user_id]
        if user_id not in self.user_id_map:
            self._action_cache[user_id] = []
            return []
        scores = self._score_all_movies_for_user(self.user_id_map[user_id])
        top_idx = np.argpartition(-scores, self.top_k - 1)[: self.top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        candidates = [int(self.idx_to_movie_id[i]) for i in top_idx]
        self._action_cache[user_id] = candidates
        return candidates

    def step(self, user_id, movie_id):
        """Reward for recommending movie_id to user_id: 1 if historical data
        shows the user rated that movie >= 4.0, else 0."""
        key = user_id * self._movie_id_bound + movie_id
        idx = np.searchsorted(self._reward_keys, key)
        if idx < len(self._reward_keys) and self._reward_keys[idx] == key:
            return int(self._reward_values[idx])
        return 0

    def most_popular_movie(self):
        """Fixed baseline action: the single most-rated movie in the whole
        catalog, recommended to every user every round."""
        return self._most_popular_movie_id

    def random_movie(self, rng):
        """Naive baseline action: a uniformly random movie from the full
        catalog, drawn fresh each round."""
        return int(rng.choice(self.catalog_movie_ids))
