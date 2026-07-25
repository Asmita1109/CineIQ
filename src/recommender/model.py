"""Neural Collaborative Filtering (NCF) model and BPR training datasets for
CineIQ's recommendation engine -- Version 3.

Version 3 moves from pointwise rating prediction (MSE against star ratings,
sigmoid-scaled to 0.5-5.0) to pairwise ranking (BPR loss: push the score for
a movie a user actually rated above the score for movies sampled as
negatives). NCF now outputs a single raw relevance score with no scaling --
only relative order between a positive and its negatives matters, not the
absolute value.

CineIQDataset/CineIQIterableDataset (the old pointwise-rating loaders) have
been replaced by CineIQBPRDataset/CineIQBPRIterableDataset, which yield
(user, positive_movie, negative_movies) rather than (user, movie, rating).
Nothing else in this codebase depended on the old classes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset


class NCF(nn.Module):
    """User embedding + movie embedding + genome embedding -> MLP -> raw
    relevance score (higher = more relevant; unbounded, not a rating).

    Default dims: embedding_dim=64 (user) + embedding_dim=64 (movie) +
    genome_dim=50 = 178-dim MLP input.
    """

    def __init__(self, num_users, num_movies, embedding_dim=64, genome_dim=50, hidden_layers=(128, 64, 32), dropout=0.2):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        input_dim = embedding_dim * 2 + genome_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, movie_ids, genome_embedding):
        user_vec = self.user_embedding(user_ids)
        movie_vec = self.movie_embedding(movie_ids)
        x = torch.cat([user_vec, movie_vec, genome_embedding], dim=-1)
        return self.mlp(x).squeeze(-1)


# ------------------------------------------------------------------
# Shared helpers used by both BPR dataset variants
# ------------------------------------------------------------------
def _build_id_maps_and_genome(user_features_path, movie_features_path, user_id_map, movie_id_map):
    user_ids_full = pd.read_parquet(user_features_path, columns=["userId"])["userId"]
    movie_features = pd.read_parquet(movie_features_path)

    if user_id_map is None:
        user_id_map = {uid: i for i, uid in enumerate(sorted(user_ids_full.unique()))}
    if movie_id_map is None:
        movie_id_map = {mid: i for i, mid in enumerate(sorted(movie_features["movieId"].unique()))}

    genome_cols = [c for c in movie_features.columns if c.startswith("genome_emb_")]
    genome_dim = len(genome_cols)

    ids_in_order = [None] * len(movie_id_map)
    for movie_id, idx in movie_id_map.items():
        ids_in_order[idx] = movie_id
    genome_lookup = (
        movie_features.set_index("movieId")[genome_cols]
        .reindex(ids_in_order)
        .fillna(0.0)
        .to_numpy(dtype="float32")
    )

    return user_id_map, movie_id_map, genome_lookup, genome_dim


def _build_user_rated_sets(history_path, user_id_map, movie_id_map):
    """userId -> frozenset of mapped movie indices they've ever rated, so
    negative sampling never draws something the user actually likes.

    history_path may be a single parquet path (e.g. rl_features.parquet, the
    fullest available interaction log) or a list of paths (e.g.
    [train_path, val_path], for contexts like a SageMaker training container
    where only specific channeled files are available) -- either way, every
    row from every path is unioned before building the per-user sets.
    """
    paths = [history_path] if isinstance(history_path, (str, Path)) else list(history_path)
    frames = [pd.read_parquet(p, columns=["userId", "movieId"]) for p in paths]
    history = pd.concat(frames, ignore_index=True).drop_duplicates() if len(frames) > 1 else frames[0]

    history = history[history["userId"].isin(user_id_map) & history["movieId"].isin(movie_id_map)]
    history = history.assign(
        user_idx=history["userId"].map(user_id_map),
        movie_idx=history["movieId"].map(movie_id_map),
    )
    return history.groupby("user_idx")["movie_idx"].apply(lambda s: frozenset(s.to_numpy())).to_dict()


class CineIQBPRDataset(Dataset):
    """Map-style BPR dataset: loads every positive interaction from
    interactions_path into memory, and for each one samples `neg_samples`
    fresh negative movies (drawn from movies the user has never rated,
    across history_path) every time that item is accessed. Since a
    DataLoader with shuffle=True re-iterates the dataset each epoch, this
    means negatives are naturally resampled every epoch -- standard BPR
    practice, rather than fixing one negative set for all of training.

    Each item is (user_idx, pos_movie_idx, pos_genome, neg_movie_idx,
    neg_genome), where neg_movie_idx/neg_genome carry `neg_samples` entries.
    """

    def __init__(
        self,
        interactions_path,
        user_features_path,
        movie_features_path,
        rated_history_path=None,
        user_id_map=None,
        movie_id_map=None,
        neg_samples=4,
        seed=42,
    ):
        self.user_id_map, self.movie_id_map, self.genome_lookup, self.genome_dim = _build_id_maps_and_genome(
            user_features_path, movie_features_path, user_id_map, movie_id_map
        )
        self.num_users = len(self.user_id_map)
        self.num_movies = len(self.movie_id_map)
        self.neg_samples = neg_samples
        self.rng = np.random.default_rng(seed)

        history_path = rated_history_path if rated_history_path is not None else interactions_path
        self.user_rated_movies = _build_user_rated_sets(history_path, self.user_id_map, self.movie_id_map)

        interactions = pd.read_parquet(interactions_path, columns=["userId", "movieId"])
        interactions = interactions[
            interactions["userId"].isin(self.user_id_map) & interactions["movieId"].isin(self.movie_id_map)
        ]
        self.user_idx = interactions["userId"].map(self.user_id_map).to_numpy(dtype="int64")
        self.pos_movie_idx = interactions["movieId"].map(self.movie_id_map).to_numpy(dtype="int64")

    def __len__(self):
        return len(self.user_idx)

    def _sample_negatives(self, user_idx):
        rated = self.user_rated_movies.get(user_idx, frozenset())
        result = []
        while len(result) < self.neg_samples:
            draws = self.rng.integers(0, self.num_movies, size=self.neg_samples - len(result))
            for d in draws:
                if d not in rated and len(result) < self.neg_samples:
                    result.append(int(d))
        return np.array(result, dtype="int64")

    def __getitem__(self, idx):
        user_idx = self.user_idx[idx]
        pos_movie_idx = self.pos_movie_idx[idx]
        neg_movie_idx = self._sample_negatives(user_idx)

        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(pos_movie_idx, dtype=torch.long),
            torch.from_numpy(self.genome_lookup[pos_movie_idx]),
            torch.from_numpy(neg_movie_idx),
            torch.from_numpy(self.genome_lookup[neg_movie_idx]),
        )


class CineIQBPRIterableDataset(IterableDataset):
    """Streaming BPR dataset: reads interactions_path 500K rows at a time via
    pyarrow (for memory-constrained training instances) instead of loading
    the full positive-interaction log into memory, sampling `neg_samples`
    negatives per positive within each chunk. Yields already-batched tensors
    directly -- construct its DataLoader with batch_size=None, same
    convention as the old CineIQIterableDataset.
    """

    def __init__(
        self,
        interactions_path,
        user_features_path,
        movie_features_path,
        rated_history_path=None,
        user_id_map=None,
        movie_id_map=None,
        neg_samples=4,
        chunk_size=500_000,
        batch_size=1024,
        seed=42,
    ):
        self.interactions_path = str(interactions_path)
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.neg_samples = neg_samples
        self.seed = seed

        self.user_id_map, self.movie_id_map, self.genome_lookup, self.genome_dim = _build_id_maps_and_genome(
            user_features_path, movie_features_path, user_id_map, movie_id_map
        )
        self.num_users = len(self.user_id_map)
        self.num_movies = len(self.movie_id_map)

        history_path = rated_history_path if rated_history_path is not None else interactions_path
        self.user_rated_movies = _build_user_rated_sets(history_path, self.user_id_map, self.movie_id_map)

        self._num_rows = pq.ParquetFile(self.interactions_path).metadata.num_rows

    def __len__(self):
        return self._num_rows

    def _sample_negatives_batch(self, user_idx_batch, rng):
        n = len(user_idx_batch)
        k = self.neg_samples
        neg = np.empty((n, k), dtype="int64")
        for i in range(n):
            rated = self.user_rated_movies.get(user_idx_batch[i], frozenset())
            count = 0
            while count < k:
                draw = int(rng.integers(0, self.num_movies))
                if draw not in rated:
                    neg[i, count] = draw
                    count += 1
        return neg

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        parquet_file = pq.ParquetFile(self.interactions_path)
        pending = None  # leftover rows carried across chunk boundaries so batches stay exactly batch_size

        for batch in parquet_file.iter_batches(batch_size=self.chunk_size, columns=["userId", "movieId"]):
            chunk = batch.to_pandas()
            chunk = chunk[
                chunk["userId"].isin(self.user_id_map) & chunk["movieId"].isin(self.movie_id_map)
            ]
            user_idx = chunk["userId"].map(self.user_id_map).to_numpy(dtype="int64")
            pos_movie_idx = chunk["movieId"].map(self.movie_id_map).to_numpy(dtype="int64")

            if pending is not None:
                p_user, p_pos = pending
                user_idx = np.concatenate([p_user, user_idx])
                pos_movie_idx = np.concatenate([p_pos, pos_movie_idx])

            n = len(user_idx)
            n_full_batches = n // self.batch_size
            for b in range(n_full_batches):
                s, e = b * self.batch_size, (b + 1) * self.batch_size
                u = user_idx[s:e]
                p = pos_movie_idx[s:e]
                neg = self._sample_negatives_batch(u, rng)

                yield (
                    torch.from_numpy(u),
                    torch.from_numpy(p),
                    torch.from_numpy(self.genome_lookup[p]),
                    torch.from_numpy(neg),
                    torch.from_numpy(self.genome_lookup[neg]),
                )

            remainder = n - n_full_batches * self.batch_size
            pending = (user_idx[-remainder:], pos_movie_idx[-remainder:]) if remainder else None

        if pending is not None and len(pending[0]) > 0:
            p_user, p_pos = pending
            neg = self._sample_negatives_batch(p_user, rng)
            yield (
                torch.from_numpy(p_user),
                torch.from_numpy(p_pos),
                torch.from_numpy(self.genome_lookup[p_pos]),
                torch.from_numpy(neg),
                torch.from_numpy(self.genome_lookup[neg]),
            )
