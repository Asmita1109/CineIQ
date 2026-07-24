"""Launch a SageMaker training job for the CineIQ NCF recommender via boto3.

Safe by default: this script ALWAYS prints the required IAM role setup
instructions, the full job configuration, and a cost estimate -- it never
mutates AWS (no code upload, no job creation) unless you explicitly pass
--role-arn (validated to actually exist) AND --launch.

Usage:
    python launch_sagemaker.py                              # instructions + dry run, no AWS writes
    python launch_sagemaker.py --role-arn <arn>              # validates the role, still dry run
    python launch_sagemaker.py --role-arn <arn> --launch     # actually submits the job
"""

import argparse
import io
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# ----------------------------------------------------------------------
# Configuration -- adjust to match your account/region/bucket
# ----------------------------------------------------------------------
REGION = "us-east-1"
S3_BUCKET = "cineiq-ml-bucket"
S3_FEATURES_PREFIX = f"s3://{S3_BUCKET}/features/"
S3_OUTPUT_PATH = f"s3://{S3_BUCKET}/models/recommender/"

INSTANCE_TYPE = "ml.m5.2xlarge"
INSTANCE_COUNT = 1
VOLUME_SIZE_GB = 30
MAX_RUNTIME_SECONDS = 24 * 60 * 60  # 24h safety cap so a stuck job can't run (and bill) forever

# AWS's Deep Learning Containers account (763104351884) is the same across
# every AWS account and most regions -- it's not specific to this project.
# Verify this exact tag is still published before relying on it:
# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
TRAINING_IMAGE = (
    f"763104351884.dkr.ecr.{REGION}.amazonaws.com/"
    "pytorch-training:2.0.0-cpu-py310-ubuntu20.04-sagemaker"
)

HYPERPARAMETERS = {
    "embedding_dim": "32",
    "lr": "0.001",
    "batch_size": "1024",
    "max_epochs": "20",
    "patience": "5",
}

# ml.m5.2xlarge on-demand SageMaker training price, us-east-1 (2x the vCPU/RAM
# of ml.m5.xlarge, and ~2x the price in the m5 family). Confirm the current
# rate at https://aws.amazon.com/sagemaker/pricing/ before relying on this
# for budgeting -- it changes and varies by region.
INSTANCE_HOURLY_USD = 0.461

SOURCE_FILES = ["model.py", "train.py", "sagemaker_train.py"]


def print_iam_role_instructions():
    print("=" * 70)
    print("IAM ROLE SETUP (required before launching)")
    print("=" * 70)
    print(
        """
SageMaker needs an execution role it can assume to run the training job.
Create it once with an IAM user that has role-creation permissions -- do
NOT use root credentials for this or for launching jobs day-to-day.

1. Trust policy (who can assume this role) -- save as trust-policy.json:

{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}

2. Create the role:

   aws iam create-role \\
     --role-name CineIQSageMakerExecutionRole \\
     --assume-role-policy-document file://trust-policy.json

3. Attach permissions. Start from the AWS-managed SageMaker policy, then
   scope S3 access to just this project's bucket instead of granting
   AmazonS3FullAccess:

   aws iam attach-role-policy \\
     --role-name CineIQSageMakerExecutionRole \\
     --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

   Then attach an inline policy granting GetObject/PutObject/ListBucket on
   arn:aws:s3:::cineiq-ml-bucket and arn:aws:s3:::cineiq-ml-bucket/* only.

4. Take the resulting role ARN (arn:aws:iam::<account-id>:role/CineIQSageMakerExecutionRole)
   and pass it to this script with --role-arn.
"""
    )


def estimate_cost(max_epochs):
    # Rough heuristic, not a quote -- actual wall-clock time depends on data
    # volume and I/O as much as epoch count.
    estimated_hours = max(0.25, 0.05 * max_epochs)
    estimated_cost = estimated_hours * INSTANCE_HOURLY_USD * INSTANCE_COUNT
    print("\n" + "=" * 70)
    print("ESTIMATED COST")
    print("=" * 70)
    print(f"Instance: {INSTANCE_TYPE} x{INSTANCE_COUNT} @ ${INSTANCE_HOURLY_USD:.3f}/hr (on-demand, {REGION})")
    print(
        f"Rough estimated runtime: ~{estimated_hours:.2f} hr "
        f"(heuristic based on {max_epochs} epochs -- treat as a ballpark, not a quote)"
    )
    print(f"Rough estimated compute cost: ~${estimated_cost:.2f}")
    print("(SageMaker separately bills storage/data transfer; this covers compute only.)")
    print("Confirm current pricing at https://aws.amazon.com/sagemaker/pricing/")


def package_and_upload_source(job_name, region):
    source_dir = Path(__file__).resolve().parent
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for filename in SOURCE_FILES:
            tar.add(source_dir / filename, arcname=filename)
    buffer.seek(0)

    s3 = boto3.client("s3", region_name=region)
    key = f"code/{job_name}/source.tar.gz"
    s3.upload_fileobj(buffer, S3_BUCKET, key)
    s3_uri = f"s3://{S3_BUCKET}/{key}"
    print(f"  Uploaded {', '.join(SOURCE_FILES)} -> {s3_uri}")
    return s3_uri


def build_job_config(job_name, role_arn, hyperparameters):
    def channel(name, filename):
        return {
            "ChannelName": name,
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"{S3_FEATURES_PREFIX}{filename}",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        }

    return {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {"TrainingImage": TRAINING_IMAGE, "TrainingInputMode": "File"},
        "RoleArn": role_arn,
        "InputDataConfig": [
            channel("train", "rec_train.parquet"),
            channel("val", "rec_val.parquet"),
            channel("user_features", "user_features.parquet"),
            channel("movie_features", "movie_features.parquet"),
        ],
        "OutputDataConfig": {"S3OutputPath": S3_OUTPUT_PATH},
        "ResourceConfig": {
            "InstanceType": INSTANCE_TYPE,
            "InstanceCount": INSTANCE_COUNT,
            "VolumeSizeInGB": VOLUME_SIZE_GB,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": MAX_RUNTIME_SECONDS},
        "HyperParameters": hyperparameters,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", type=str, default=None, help="Existing SageMaker execution role ARN")
    parser.add_argument("--launch", action="store_true", help="Actually submit the job (default: dry run)")
    parser.add_argument("--region", type=str, default=REGION)
    args = parser.parse_args()

    print_iam_role_instructions()

    job_name = f"cineiq-recommender-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    console_link = (
        f"https://{args.region}.console.aws.amazon.com/sagemaker/home"
        f"?region={args.region}#/jobs/{job_name}"
    )

    print("\n" + "=" * 70)
    print("JOB CONFIGURATION")
    print("=" * 70)
    print(f"Job name:        {job_name}")
    print(f"Region:          {args.region}")
    print(f"Training image:  {TRAINING_IMAGE}")
    print(f"Instance:        {INSTANCE_TYPE} x{INSTANCE_COUNT}, {VOLUME_SIZE_GB}GB volume")
    print(f"Input channels:  train, val, user_features, movie_features -> {S3_FEATURES_PREFIX}")
    print(f"Output path:     {S3_OUTPUT_PATH}")
    print("Hyperparameters:")
    for k, v in HYPERPARAMETERS.items():
        print(f"  {k} = {v}")

    estimate_cost(int(HYPERPARAMETERS["max_epochs"]))

    print(f"\nSageMaker console link (once launched): {console_link}")

    if not args.role_arn:
        print("\n" + "=" * 70)
        print("NOT LAUNCHING")
        print("=" * 70)
        print("No --role-arn provided. Set up the IAM role above, then re-run with:")
        print(f"  python {Path(__file__).name} --role-arn <role-arn> --launch")
        return

    iam = boto3.client("iam", region_name=args.region)
    role_name = args.role_arn.split("/")[-1]
    try:
        iam.get_role(RoleName=role_name)
        print(f"\nRole check: {args.role_arn} exists.")
    except ClientError as e:
        print(f"\nRole check FAILED: {e}")
        print("Fix the role before launching. Not submitting the job.")
        return

    if not args.launch:
        print("\n--role-arn was valid but --launch was not passed. Dry run only -- not submitting the job.")
        print(f"Re-run with --launch to actually create the training job {job_name}.")
        return

    print("\nPackaging and uploading source code...")
    submit_dir = package_and_upload_source(job_name, args.region)

    hyperparameters = dict(HYPERPARAMETERS)
    hyperparameters["sagemaker_program"] = "sagemaker_train.py"
    hyperparameters["sagemaker_submit_directory"] = submit_dir

    sm = boto3.client("sagemaker", region_name=args.region)
    config = build_job_config(job_name, args.role_arn, hyperparameters)

    print("\nSubmitting training job...")
    response = sm.create_training_job(**config)
    print(f"Launched: {response['TrainingJobArn']}")
    print(f"Monitor: {console_link}")


if __name__ == "__main__":
    main()
