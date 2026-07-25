"""SageMaker entry point for training the CineIQ NCF recommender -- Version 3
(BPR pairwise ranking loss).

Wraps train.train_model() so the exact same model/loss/optimizer/eval logic
runs locally and in a SageMaker training job -- only the data loading
strategy, data/model paths, and hyperparameter source change. Deploy by
pointing a SageMaker Estimator's source_dir at src/recommender/ (so model.py,
train.py, and evaluate.py all ship alongside this script -- train.py imports
ranking-metric helpers from evaluate.py) and entry_point at
"sagemaker_train.py".

Uses CineIQBPRIterableDataset (chunked pyarrow reads, 500K rows at a time)
for rec_train.parquet instead of train.py's default CineIQBPRDataset (loads
all positive interactions into memory), since the full in-memory load can
OOM a smaller training instance like ml.m5.xlarge (16GB RAM) -- see
model.CineIQBPRIterableDataset for how batching/remainders/negative sampling
work per chunk.

Negative sampling excludes each user's full rating history when possible,
but rl_features.parquet (the fullest interaction log) isn't one of the
channeled files here -- train_model() detects it's absent in this container
and falls back to excluding whatever the train + val channels alone show,
which is what train_model()'s rated_history_path=None default already does
automatically; no extra channel is needed.

Expects four data channels passed to Estimator.fit({...}) (matches
launch_sagemaker.py's four-channel job config):
  train          -> rec_train.parquet
  val            -> rec_val.parquet
  user_features  -> user_features.parquet
  movie_features -> movie_features.parquet
"""

import argparse
import os
from pathlib import Path

from model import CineIQBPRIterableDataset
from train import train_model

# SageMaker conventions: each channel passed to Estimator.fit() is mounted at
# /opt/ml/input/data/<channel> and exposed via an SM_CHANNEL_<NAME> env var;
# the model artifact directory (tarred up and pushed to S3 after training) is
# /opt/ml/model, exposed via SM_MODEL_DIR.
SM_INPUT_DIR = Path("/opt/ml/input/data")
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters -- SageMaker passes Estimator(hyperparameters={...}) as
    # CLI args to the entry point script.
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--hidden_layers", type=str, default="128,64,32")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--neg_samples", type=int, default=4, help="Negatives sampled per positive interaction")
    parser.add_argument(
        "--val_neg_samples", type=int, default=100, help="Negatives sampled per user for validation ranking metrics"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=500_000, help="Rows read per pyarrow batch while streaming"
    )

    # Data channels -- default to the SM_CHANNEL_* env vars SageMaker sets;
    # fall back to the raw /opt/ml/input/data/<channel> path so this still
    # works without the SageMaker SDK setting those env vars (e.g. a plain
    # `docker run` against the training image for local testing).
    parser.add_argument(
        "--train_channel", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", str(SM_INPUT_DIR / "train"))
    )
    parser.add_argument(
        "--val_channel", type=str, default=os.environ.get("SM_CHANNEL_VAL", str(SM_INPUT_DIR / "val"))
    )
    parser.add_argument(
        "--user_features_channel",
        type=str,
        default=os.environ.get("SM_CHANNEL_USER_FEATURES", str(SM_INPUT_DIR / "user_features")),
    )
    parser.add_argument(
        "--movie_features_channel",
        type=str,
        default=os.environ.get("SM_CHANNEL_MOVIE_FEATURES", str(SM_INPUT_DIR / "movie_features")),
    )
    parser.add_argument("--model_dir", type=str, default=SM_MODEL_DIR)

    return parser.parse_args()


def main():
    args = parse_args()
    hidden_layers = tuple(int(x) for x in args.hidden_layers.split(","))

    train_channel = Path(args.train_channel)
    val_channel = Path(args.val_channel)
    user_features_channel = Path(args.user_features_channel)
    movie_features_channel = Path(args.movie_features_channel)
    model_dir = Path(args.model_dir)

    print(f"train_channel:          {train_channel}")
    print(f"val_channel:            {val_channel}")
    print(f"user_features_channel:  {user_features_channel}")
    print(f"movie_features_channel: {movie_features_channel}")
    print(f"model_dir:              {model_dir}")
    print(
        f"hyperparameters:  embedding_dim={args.embedding_dim}  lr={args.lr}  "
        f"batch_size={args.batch_size}  hidden_layers={hidden_layers}  dropout={args.dropout}  "
        f"neg_samples={args.neg_samples}  val_neg_samples={args.val_neg_samples}  "
        f"max_epochs={args.max_epochs}  patience={args.patience}  chunk_size={args.chunk_size}"
    )

    train_model(
        train_path=train_channel / "rec_train.parquet",
        val_path=val_channel / "rec_val.parquet",
        user_features_path=user_features_channel / "user_features.parquet",
        movie_features_path=movie_features_channel / "movie_features.parquet",
        model_output_path=model_dir / "recommender_model.pt",
        figures_dir=None,  # no figures channel in the training container by default
        embedding_dim=args.embedding_dim,
        hidden_layers=hidden_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        neg_samples=args.neg_samples,
        val_neg_samples=args.val_neg_samples,
        dataset_class=CineIQBPRIterableDataset,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
