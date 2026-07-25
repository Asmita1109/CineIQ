"""Fetch epoch-level training metrics for a SageMaker training job.

The job wasn't launched with MetricDefinitions configured (launch_sagemaker.py's
AlgorithmSpecification doesn't set any), so SageMaker's structured metrics API
(DescribeTrainingJob's FinalMetricDataList / the CloudWatch custom metrics
namespace) has nothing to return. This instead parses the same
"Epoch N | train MSE: ... | val RMSE: ... | val NDCG@10: ..." lines
train_model() already prints, straight out of the job's CloudWatch Logs.
"""

import json
import re
from pathlib import Path

import boto3

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

JOB_NAME = "cineiq-recommender-20260724-175332"
REGION = "us-east-1"
LOG_GROUP = "/aws/sagemaker/TrainingJobs"

EPOCH_LINE_RE = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*train MSE:\s*([\d.]+)\s*\|\s*val RMSE:\s*([\d.]+)\s*\|\s*val NDCG@10:\s*([\d.]+)"
)


def get_log_stream_names(logs_client, job_name):
    resp = logs_client.describe_log_streams(
        logGroupName=LOG_GROUP,
        logStreamNamePrefix=f"{job_name}/",
    )
    streams = [s["logStreamName"] for s in resp.get("logStreams", [])]
    if not streams:
        raise RuntimeError(f"No CloudWatch log streams found for job {job_name} under {LOG_GROUP}")
    return streams


def fetch_all_log_lines(logs_client, log_stream_name):
    lines = []
    next_token = None
    while True:
        kwargs = {"logGroupName": LOG_GROUP, "logStreamName": log_stream_name, "startFromHead": True}
        if next_token:
            kwargs["nextToken"] = next_token
        resp = logs_client.get_log_events(**kwargs)
        events = resp.get("events", [])
        lines.extend(e["message"] for e in events)
        new_token = resp.get("nextForwardToken")
        if new_token == next_token or not events:
            break
        next_token = new_token
    return lines


def parse_epoch_metrics(lines):
    epochs = []
    for line in lines:
        m = EPOCH_LINE_RE.search(line)
        if m:
            epochs.append(
                {
                    "epoch": int(m.group(1)),
                    "train_mse": float(m.group(2)),
                    "val_rmse": float(m.group(3)),
                    "val_ndcg_10": float(m.group(4)),
                }
            )
    return epochs


def main():
    sm = boto3.client("sagemaker", region_name=REGION)
    logs_client = boto3.client("logs", region_name=REGION)

    print(f"Fetching training job details: {JOB_NAME}")
    job = sm.describe_training_job(TrainingJobName=JOB_NAME)
    print(f"  Status: {job['TrainingJobStatus']} / {job.get('SecondaryStatus')}")
    print(f"  Instance: {job['ResourceConfig']['InstanceType']}")

    stream_names = get_log_stream_names(logs_client, JOB_NAME)
    print(f"  Log streams found: {stream_names}")

    all_lines = []
    for stream_name in stream_names:
        all_lines.extend(fetch_all_log_lines(logs_client, stream_name))
    print(f"  Total log lines fetched: {len(all_lines):,}")

    epochs = parse_epoch_metrics(all_lines)
    epochs.sort(key=lambda e: e["epoch"])

    if not epochs:
        print(
            "\nNo epoch metric lines found in the logs. The job may not have logged any "
            "epochs yet, or the log line format didn't match."
        )
        return

    print("\n" + "=" * 60)
    print(f"{'Epoch':>6} | {'Train MSE':>10} | {'Val RMSE':>10} | {'Val NDCG@10':>12}")
    print("=" * 60)
    for e in epochs:
        print(f"{e['epoch']:>6} | {e['train_mse']:>10.4f} | {e['val_rmse']:>10.4f} | {e['val_ndcg_10']:>12.4f}")
    print("=" * 60)
    print(f"Total epochs found: {len(epochs)}")

    best_epoch = min(epochs, key=lambda e: e["val_rmse"])
    print(
        f"\nBest epoch by val RMSE: {best_epoch['epoch']} "
        f"(val RMSE {best_epoch['val_rmse']:.4f}, val NDCG@10 {best_epoch['val_ndcg_10']:.4f})"
    )

    results = {
        "job_name": JOB_NAME,
        "job_status": job["TrainingJobStatus"],
        "secondary_status": job.get("SecondaryStatus"),
        "instance_type": job["ResourceConfig"]["InstanceType"],
        "hyperparameters": job["HyperParameters"],
        "epochs": epochs,
        "best_epoch": best_epoch,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / "training_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    main()
