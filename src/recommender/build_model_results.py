"""Extract key metrics from the training/evaluation result artifacts in
models/ into a single results/model_results.json summary.

Sources:
  models/forecasting_results.json          -- LightGBM forecasting train/val/test run
  models/training_results.json             -- NCF (V2, pointwise) SageMaker training run
  models/recommender_test_results_bpr.json -- NCF (V3, BPR) test-set evaluation
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

FORECASTING_RESULTS_PATH = MODELS_DIR / "forecasting_results.json"
TRAINING_RESULTS_PATH = MODELS_DIR / "training_results.json"
BPR_TEST_RESULTS_PATH = MODELS_DIR / "recommender_test_results_bpr.json"
OUTPUT_PATH = RESULTS_DIR / "model_results.json"


def extract_forecasting(forecasting_results):
    return {
        "best_iteration": forecasting_results["best_iteration"],
        "n_rows": forecasting_results["n_rows"],
        "val": forecasting_results["val"],
        "test": forecasting_results["test"],
    }


def extract_ncf_v2_training(training_results):
    best = training_results["best_epoch"]
    return {
        "job_name": training_results.get("job_name"),
        "job_status": training_results.get("job_status"),
        "hyperparameters": training_results.get("hyperparameters", {}),
        "n_epochs_logged": len(training_results.get("epochs", [])),
        "best_epoch": {
            "epoch": best["epoch"],
            "train_mse": best["train_mse"],
            "val_rmse": best["val_rmse"],
            "val_ndcg_10": best["val_ndcg_10"],
        },
    }


def extract_ncf_v3_bpr_test(bpr_results):
    return {
        "checkpoint_epoch": bpr_results["checkpoint_epoch"],
        "val_metrics": {
            "ndcg_10": bpr_results["checkpoint_val_ndcg_10"],
            "precision_10": bpr_results["checkpoint_val_precision_10"],
            "recall_10": bpr_results["checkpoint_val_recall_10"],
        },
        "test_metrics": bpr_results["test_metrics"],
        "methodology": bpr_results.get("methodology", {}),
        "ranking_metrics_by_user_segment": bpr_results.get("ranking_metrics_by_user_segment", {}),
        "ndcg_10_by_top_genres": bpr_results.get("ndcg_10_by_top_genres", {}),
    }


def main():
    with open(FORECASTING_RESULTS_PATH) as f:
        forecasting_results = json.load(f)
    with open(TRAINING_RESULTS_PATH) as f:
        training_results = json.load(f)
    with open(BPR_TEST_RESULTS_PATH) as f:
        bpr_results = json.load(f)

    summary = {
        "forecasting": extract_forecasting(forecasting_results),
        "recommender": {
            "ncf_v2_pointwise_training": extract_ncf_v2_training(training_results),
            "ncf_v3_bpr_test_evaluation": extract_ncf_v3_bpr_test(bpr_results),
        },
        "sources": {
            "forecasting_results": str(FORECASTING_RESULTS_PATH.relative_to(PROJECT_ROOT)),
            "training_results": str(TRAINING_RESULTS_PATH.relative_to(PROJECT_ROOT)),
            "bpr_test_results": str(BPR_TEST_RESULTS_PATH.relative_to(PROJECT_ROOT)),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
