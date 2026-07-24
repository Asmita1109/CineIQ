"""Neural Collaborative Filtering (NCF) model and dataset for CineIQ's
recommendation engine."""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset

RATING_MIN = 0.5
RATING_MAX = 5.0


class NCF(nn.Module):
    """User embedding + movie embedding + genome embedding -> MLP -> rating.

    Default dims: embedding_dim=32 (user) + embedding_dim=32 (movie) +
    genome_dim=50 = 114-dim MLP input.
    """

    def __init__(self, num_users, num_movies, embedding_dim=32, genome_dim=50, hidden_layers=(128, 64, 32), dropout=0.2):
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
        logits = self.mlp(x).squeeze(-1)
        rating = torch.sigmoid(logits) * (RATING_MAX - RATING_MIN) + RATING_MIN
        return rating


class CineIQDataset(Dataset):
    """Joins a rec_*.parquet interaction log against movie_features.parquet
    (for genome embeddings) and user_features.parquet (for the full user id
    space, so embedding indices stay consistent across train/val/test splits
    even though this split may not contain every user).

    Pass user_id_map/movie_id_map explicitly (e.g. the maps built from the
    training set) when constructing val/test datasets, so the same raw
    userId/movieId always maps to the same embedding row across all splits.
    """

    def __init__(
        self,
        interactions_path,
        user_features_path,
        movie_features_path,
        user_id_map=None,
        movie_id_map=None,
        max_rows=None,
    ):
        user_ids_full = pd.read_parquet(user_features_path, columns=["userId"])["userId"]
        movie_features = pd.read_parquet(movie_features_path)

        if user_id_map is None:
            user_id_map = {uid: i for i, uid in enumerate(sorted(user_ids_full.unique()))}
        if movie_id_map is None:
            movie_id_map = {mid: i for i, mid in enumerate(sorted(movie_features["movieId"].unique()))}

        self.user_id_map = user_id_map
        self.movie_id_map = movie_id_map
        self.num_users = len(user_id_map)
        self.num_movies = len(movie_id_map)

        genome_cols = [c for c in movie_features.columns if c.startswith("genome_emb_")]
        self.genome_dim = len(genome_cols)
        movie_genome = movie_features[["movieId"] + genome_cols]

        interactions = pd.read_parquet(interactions_path, columns=["userId", "movieId", "rating"])
        if max_rows is not None:
            interactions = interactions.head(max_rows)
        interactions = interactions[
            interactions["userId"].isin(self.user_id_map) & interactions["movieId"].isin(self.movie_id_map)
        ]
        interactions = interactions.merge(movie_genome, on="movieId", how="left")
        interactions[genome_cols] = interactions[genome_cols].fillna(0.0)

        self.user_idx = torch.tensor(
            interactions["userId"].map(self.user_id_map).to_numpy(dtype="int64"), dtype=torch.long
        )
        self.movie_idx = torch.tensor(
            interactions["movieId"].map(self.movie_id_map).to_numpy(dtype="int64"), dtype=torch.long
        )
        self.genome = torch.tensor(interactions[genome_cols].to_numpy(dtype="float32"), dtype=torch.float32)
        self.rating = torch.tensor(interactions["rating"].to_numpy(dtype="float32"), dtype=torch.float32)

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        return self.user_idx[idx], self.movie_idx[idx], self.genome[idx], self.rating[idx]


class CineIQIterableDataset(IterableDataset):
    """Streams a rec_*.parquet interaction log 500K rows at a time via
    pyarrow instead of loading it fully into memory -- for training instances
    (e.g. SageMaker ml.m5.xlarge, 16GB RAM) where the full CineIQDataset's
    in-memory join could OOM. user_features.parquet/movie_features.parquet
    are still read in full (they're small -- a few MB each) purely to build
    the same stable userId/movieId -> embedding-index maps CineIQDataset uses,
    so a model trained one way can be evaluated/served the other way.

    Yields already-batched tensors (batch_size each) directly, carrying any
    remainder over the next 500K-row read so batches stay exactly batch_size
    across chunk boundaries. Construct its DataLoader with batch_size=None --
    this dataset does its own batching, and IterableDataset doesn't support
    DataLoader's shuffle=True (reads sequentially from disk).
    """

    def __init__(
        self,
        interactions_path,
        user_features_path,
        movie_features_path,
        user_id_map=None,
        movie_id_map=None,
        chunk_size=500_000,
        batch_size=1024,
    ):
        self.interactions_path = str(interactions_path)
        self.chunk_size = chunk_size
        self.batch_size = batch_size

        user_ids_full = pd.read_parquet(user_features_path, columns=["userId"])["userId"]
        movie_features = pd.read_parquet(movie_features_path)

        if user_id_map is None:
            user_id_map = {uid: i for i, uid in enumerate(sorted(user_ids_full.unique()))}
        if movie_id_map is None:
            movie_id_map = {mid: i for i, mid in enumerate(sorted(movie_features["movieId"].unique()))}

        self.user_id_map = user_id_map
        self.movie_id_map = movie_id_map
        self.num_users = len(user_id_map)
        self.num_movies = len(movie_id_map)

        self.genome_cols = [c for c in movie_features.columns if c.startswith("genome_emb_")]
        self.genome_dim = len(self.genome_cols)
        self.movie_genome = movie_features[["movieId"] + self.genome_cols]

        # Exact row count from parquet footer metadata -- doesn't read any row data.
        self._num_rows = pq.ParquetFile(self.interactions_path).metadata.num_rows

    def __len__(self):
        return self._num_rows

    def _iter_row_chunks(self):
        parquet_file = pq.ParquetFile(self.interactions_path)
        for batch in parquet_file.iter_batches(
            batch_size=self.chunk_size, columns=["userId", "movieId", "rating"]
        ):
            chunk = batch.to_pandas()
            chunk = chunk[
                chunk["userId"].isin(self.user_id_map) & chunk["movieId"].isin(self.movie_id_map)
            ]
            chunk = chunk.merge(self.movie_genome, on="movieId", how="left")
            chunk[self.genome_cols] = chunk[self.genome_cols].fillna(0.0)

            user_idx = chunk["userId"].map(self.user_id_map).to_numpy(dtype="int64")
            movie_idx = chunk["movieId"].map(self.movie_id_map).to_numpy(dtype="int64")
            genome = chunk[self.genome_cols].to_numpy(dtype="float32")
            rating = chunk["rating"].to_numpy(dtype="float32")
            yield user_idx, movie_idx, genome, rating

    def __iter__(self):
        pending = None  # (user_idx, movie_idx, genome, rating) left over from the last chunk

        for user_idx, movie_idx, genome, rating in self._iter_row_chunks():
            if pending is not None:
                p_user, p_movie, p_genome, p_rating = pending
                user_idx = np.concatenate([p_user, user_idx])
                movie_idx = np.concatenate([p_movie, movie_idx])
                genome = np.concatenate([p_genome, genome])
                rating = np.concatenate([p_rating, rating])

            n = len(rating)
            n_full_batches = n // self.batch_size
            for b in range(n_full_batches):
                s, e = b * self.batch_size, (b + 1) * self.batch_size
                yield (
                    torch.from_numpy(user_idx[s:e]),
                    torch.from_numpy(movie_idx[s:e]),
                    torch.from_numpy(genome[s:e]),
                    torch.from_numpy(rating[s:e]),
                )

            remainder = n - n_full_batches * self.batch_size
            pending = (
                (user_idx[-remainder:], movie_idx[-remainder:], genome[-remainder:], rating[-remainder:])
                if remainder
                else None
            )

        if pending is not None and len(pending[3]) > 0:
            p_user, p_movie, p_genome, p_rating = pending
            yield (
                torch.from_numpy(p_user),
                torch.from_numpy(p_movie),
                torch.from_numpy(p_genome),
                torch.from_numpy(p_rating),
            )
