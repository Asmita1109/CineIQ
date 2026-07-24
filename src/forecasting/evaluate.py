"""Evaluate the trained forecasting model on the held-out test set and
compare against validation performance to check for overfitting."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"

# From the training run (src/forecasting/train.py), for comparison.
VAL_BASELINE_RMSE = 1272.386
VAL_BASELINE_MAE = 738.763
VAL_MODEL_RMSE = 271.185
VAL_MODEL_MAE = 149.959


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(y_true, y_pred, label):
    r = rmse(y_true, y_pred)
    m = mean_absolute_error(y_true, y_pred)
    print(f"{label}: RMSE = {r:.3f}   MAE = {m:.3f}")
    return r, m


def main():
    print(f"Loading model from {MODELS_DIR / 'forecasting_model.pkl'}")
    with open(MODELS_DIR / "forecasting_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    genre_encoder = bundle["genre_encoder"]
    feature_cols = bundle["feature_cols"]

    test = pd.read_csv(FEATURES_DIR / "forecasting_test.csv")
    print(f"forecasting_test.csv: {test.shape}")

    test["genre_encoded"] = genre_encoder.transform(test["genre"])
    X_test, y_test = test[feature_cols], test["rating_count"]

    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    baseline_preds = test["lag_1"].fillna(0)
    baseline_rmse, baseline_mae = evaluate(y_test, baseline_preds, "Baseline (lag_1)")

    model_preds = model.predict(X_test, num_iteration=model.best_iteration)
    model_preds = np.clip(model_preds, 0, None)
    model_rmse, model_mae = evaluate(y_test, model_preds, "LightGBM")

    rmse_improvement = (baseline_rmse - model_rmse) / baseline_rmse * 100
    mae_improvement = (baseline_mae - model_mae) / baseline_mae * 100

    print("\n" + "=" * 70)
    print("TEST vs VALIDATION -- GENERALIZATION CHECK")
    print("=" * 70)
    comparison = pd.DataFrame(
        {
            "Split": ["Validation", "Test"],
            "Baseline RMSE": [round(VAL_BASELINE_RMSE, 3), round(baseline_rmse, 3)],
            "Baseline MAE": [round(VAL_BASELINE_MAE, 3), round(baseline_mae, 3)],
            "LightGBM RMSE": [round(VAL_MODEL_RMSE, 3), round(model_rmse, 3)],
            "LightGBM MAE": [round(VAL_MODEL_MAE, 3), round(model_mae, 3)],
        }
    )
    print(comparison.to_string(index=False))

    rmse_gap_pct = (model_rmse - VAL_MODEL_RMSE) / VAL_MODEL_RMSE * 100
    mae_gap_pct = (model_mae - VAL_MODEL_MAE) / VAL_MODEL_MAE * 100
    print(f"\nTest vs val RMSE gap: {rmse_gap_pct:+.1f}%  (positive = test worse than val)")
    print(f"Test vs val MAE gap:  {mae_gap_pct:+.1f}%  (positive = test worse than val)")

    # Overfitting shows up as test performing WORSE than validation. Test
    # matching or beating validation is fine regardless of how large that
    # gap is -- only a worse-than-tolerance test score is a red flag.
    OVERFIT_TOLERANCE_PCT = 15
    if rmse_gap_pct <= 0:
        print(
            f"-> Test RMSE is as good as or better than validation RMSE ({rmse_gap_pct:+.1f}%): "
            f"no sign of overfitting."
        )
    elif rmse_gap_pct <= OVERFIT_TOLERANCE_PCT:
        print(
            f"-> Test RMSE is worse than validation by {rmse_gap_pct:.1f}%, within the "
            f"{OVERFIT_TOLERANCE_PCT}% tolerance: no strong sign of overfitting."
        )
    else:
        print(
            f"-> Test RMSE is worse than validation by {rmse_gap_pct:.1f}%, beyond the "
            f"{OVERFIT_TOLERANCE_PCT}% tolerance: possible overfitting, worth a closer look."
        )

    print(
        f"\nRelative improvement over baseline -- val: {(VAL_BASELINE_RMSE - VAL_MODEL_RMSE) / VAL_BASELINE_RMSE * 100:.1f}%, "
        f"test: {rmse_improvement:.1f}% (a more robust generalization signal than raw RMSE, "
        f"since it's not sensitive to each period's underlying rating volume)"
    )

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Test baseline RMSE: {baseline_rmse:.3f}")
    print(f"Test LightGBM RMSE: {model_rmse:.3f}")
    print(f"Test improvement over baseline: {rmse_improvement:+.1f}% (RMSE), {mae_improvement:+.1f}% (MAE)")


if __name__ == "__main__":
    main()
