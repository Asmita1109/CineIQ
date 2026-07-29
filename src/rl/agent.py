"""LinUCB contextual bandit agent for CineIQ's recommendation RL layer.

Classic disjoint LinUCB (Li et al. 2010): one ridge-regression linear model
per action (here, per candidate movieId), updated incrementally via the
Sherman-Morrison formula so both selection and update are O(d^2) in the
context dimension d rather than re-inverting a d x d matrix every round --
needed for this to stay feasible over tens of millions of rounds.

Context vector (see build_context) concatenates:
  - user segment, one-hot over SEGMENTS
  - the average genome embedding (movie_features.parquet) of the user's
    last 5 rated movies -- a stand-in for "recent taste"
  - a genre-trend signal: one-hot of the environment's currently trending
    genre over GENRES, plus a log-scaled trend-strength scalar
"""

import numpy as np
import pandas as pd

from environment import GENRES, SEGMENTS


class LinUCBAgent:
    def __init__(self, movie_features_path, alpha=1.0):
        self.alpha = alpha

        movie_features = pd.read_parquet(movie_features_path)
        genome_cols = [c for c in movie_features.columns if c.startswith("genome_emb_")]
        self.genome_dim = len(genome_cols)
        self.movie_genome = movie_features.set_index("movieId")[genome_cols].astype("float64")

        self.context_dim = len(SEGMENTS) + self.genome_dim + len(GENRES) + 1

        self.A_inv = {}  # action (movieId) -> d x d inverse ridge matrix
        self.b = {}  # action (movieId) -> d-dim reward-weighted context sum

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------
    def _segment_vector(self, segment):
        vec = np.zeros(len(SEGMENTS))
        if segment in SEGMENTS:
            vec[SEGMENTS.index(segment)] = 1.0
        return vec

    def _history_vector(self, recent_movies):
        embeddings = [
            self.movie_genome.loc[int(m)].to_numpy()
            for m in recent_movies
            if m is not None and not pd.isna(m) and int(m) in self.movie_genome.index
        ]
        if not embeddings:
            return np.zeros(self.genome_dim)
        return np.mean(embeddings, axis=0)

    def _trend_vector(self, trending_genre, trending_genre_score):
        vec = np.zeros(len(GENRES) + 1)
        if trending_genre in GENRES:
            vec[GENRES.index(trending_genre)] = 1.0
        vec[-1] = np.log1p(max(trending_genre_score, 0.0))
        return vec

    def build_context(self, state):
        return np.concatenate(
            [
                self._segment_vector(state["user_segment"]),
                self._history_vector(state["recent_movies"]),
                self._trend_vector(state["trending_genre"], state["trending_genre_score"]),
            ]
        )

    # ------------------------------------------------------------------
    # LinUCB core
    # ------------------------------------------------------------------
    def _ensure_arm(self, action):
        if action not in self.A_inv:
            self.A_inv[action] = np.eye(self.context_dim)
            self.b[action] = np.zeros(self.context_dim)

    def select_action(self, context, candidate_actions):
        """UCB = predicted reward + alpha * uncertainty bonus. Picks the
        candidate with the highest UCB -- alpha controls how much weight the
        uncertainty bonus (larger for actions tried less / in less-seen
        contexts) gets relative to the predicted reward, i.e. how much the
        agent explores vs. exploits."""
        best_action, best_ucb = None, -np.inf
        for action in candidate_actions:
            self._ensure_arm(action)
            A_inv = self.A_inv[action]
            theta = A_inv @ self.b[action]
            mean = float(theta @ context)
            bonus = self.alpha * float(np.sqrt(context @ A_inv @ context))
            ucb = mean + bonus
            if ucb > best_ucb:
                best_ucb, best_action = ucb, action
        return best_action

    def update(self, action, context, reward):
        """Incremental ridge-regression update for the chosen action's
        linear model, using the Sherman-Morrison identity to update A_inv
        directly (avoids re-inverting a d x d matrix every call)."""
        self._ensure_arm(action)
        A_inv = self.A_inv[action]
        Ax = A_inv @ context
        denom = 1.0 + context @ Ax
        self.A_inv[action] = A_inv - np.outer(Ax, Ax) / denom
        self.b[action] = self.b[action] + reward * context
