"""Train a LightGBM model to forecast weekly genre rating volume.

Compares against a naive "next week = last week" baseline so LightGBM has to
earn its complexity, not just beat a strawman.
"""

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = MODELS_DIR / "figures"

FEATURE_COLS = ["lag_1", "lag_2", "lag_3", "rolling_3week_avg", "rolling_6week_avg", "genre_encoded"]
TARGET_COL = "rating_count"
RANDOM_STATE = 42

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": RANDOM_STATE,
    "verbose": -1,
}
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50

PALETTE = "mako"


def load_data():
    train = pd.read_csv(FEATURES_DIR / "forecasting_train.csv")
    val = pd.read_csv(FEATURES_DIR / "forecasting_val.csv")
    test = pd.read_csv(FEATURES_DIR / "forecasting_test.csv")
    return train, val, test


def encode_genre(train, val, test):
    encoder = LabelEncoder()
    train = train.copy()
    val = val.copy()
    test = test.copy()
    train["genre_encoded"] = encoder.fit_transform(train["genre"])
    val["genre_encoded"] = encoder.transform(val["genre"])
    test["genre_encoded"] = encoder.transform(test["genre"])
    return train, val, test, encoder


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(y_true, y_pred, label):
    r = rmse(y_true, y_pred)
    m = mean_absolute_error(y_true, y_pred)
    print(f"{label}: RMSE = {r:.3f}   MAE = {m:.3f}")
    return r, m


def plot_feature_importance(model):
    importance = pd.DataFrame(
        {"feature": model.feature_name(), "importance": model.feature_importance(importance_type="gain")}
    ).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x="importance", y="feature", data=importance, hue="feature", palette=PALETTE, legend=False, ax=ax
    )
    ax.set_title("Forecasting Model -- Feature Importance (gain)")
    ax.set_xlabel("Total Gain")
    ax.set_ylabel("Feature")
    path = FIGURES_DIR / "forecasting_feature_importance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_actual_vs_predicted(y_val, val_preds):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_val, val_preds, alpha=0.3, s=15, color=sns.color_palette(PALETTE, 5)[3])
    max_val = max(float(y_val.max()), float(val_preds.max()))
    ax.plot([0, max_val], [0, max_val], color="firebrick", linestyle="--", linewidth=1.5, label="Perfect prediction")
    ax.set_title("Actual vs Predicted Weekly Rating Count (Validation)")
    ax.set_xlabel("Actual rating_count")
    ax.set_ylabel("Predicted rating_count")
    ax.legend()
    path = FIGURES_DIR / "forecasting_actual_vs_predicted.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_loss_curve(evals_result, best_iteration):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(evals_result["train"]["rmse"], label="train", color=sns.color_palette(PALETTE, 5)[3])
    ax.plot(evals_result["val"]["rmse"], label="val", color=sns.color_palette(PALETTE, 5)[1])
    ax.axvline(
        best_iteration, color="gray", linestyle="--", linewidth=1, label=f"best iteration ({best_iteration})"
    )
    ax.set_title("Training Loss Curve (RMSE)")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("RMSE")
    ax.legend()
    path = FIGURES_DIR / "forecasting_loss_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 150

    print("Loading forecasting_train.csv / forecasting_val.csv / forecasting_test.csv ...")
    train, val, test = load_data()
    print(f"  train: {train.shape}")
    print(f"  val:   {val.shape}")
    print(f"  test:  {test.shape}")

    train, val, test, genre_encoder = encode_genre(train, val, test)

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_val, y_val = val[FEATURE_COLS], val[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    # ------------------------------------------------------------------
    # 3. Naive baseline: predict next week = last week (lag_1)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BASELINE: next week = last week (lag_1)")
    print("=" * 70)
    baseline_preds = val["lag_1"].fillna(0)
    baseline_rmse, baseline_mae = evaluate(y_val, baseline_preds, "Baseline")

    # ------------------------------------------------------------------
    # 4. Train LightGBM
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING LIGHTGBM")
    print("=" * 70)
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    evals_result = {}
    model = lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(evals_result),
        ],
    )
    print(f"\nBest iteration: {model.best_iteration}")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    val_preds = np.clip(val_preds, 0, None)  # rating counts can't be negative
    model_rmse, model_mae = evaluate(y_val, val_preds, "LightGBM")

    val_rmse_improvement = (baseline_rmse - model_rmse) / baseline_rmse * 100
    val_mae_improvement = (baseline_mae - model_mae) / baseline_mae * 100

    summary = pd.DataFrame(
        {
            "Model": ["Baseline (lag_1)", "LightGBM"],
            "RMSE": [round(baseline_rmse, 3), round(model_rmse, 3)],
            "MAE": [round(baseline_mae, 3), round(model_mae, 3)],
        }
    )
    print("\n" + summary.to_string(index=False))
    print(f"\nRMSE improvement over baseline: {val_rmse_improvement:+.1f}%")
    print(f"MAE improvement over baseline:  {val_mae_improvement:+.1f}%")

    # ------------------------------------------------------------------
    # 5b. Evaluate on the held-out test set too
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    test_baseline_preds = test["lag_1"].fillna(0)
    test_baseline_rmse, test_baseline_mae = evaluate(y_test, test_baseline_preds, "Baseline (lag_1)")

    test_preds = model.predict(X_test, num_iteration=model.best_iteration)
    test_preds = np.clip(test_preds, 0, None)
    test_model_rmse, test_model_mae = evaluate(y_test, test_preds, "LightGBM")

    test_rmse_improvement = (test_baseline_rmse - test_model_rmse) / test_baseline_rmse * 100
    test_mae_improvement = (test_baseline_mae - test_model_mae) / test_baseline_mae * 100
    print(f"\nRMSE improvement over baseline: {test_rmse_improvement:+.1f}%")
    print(f"MAE improvement over baseline:  {test_mae_improvement:+.1f}%")

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    plot_feature_importance(model)
    plot_actual_vs_predicted(y_val, val_preds)
    plot_loss_curve(evals_result, model.best_iteration)

    # ------------------------------------------------------------------
    # 7. Save model
    # ------------------------------------------------------------------
    model_path = MODELS_DIR / "forecasting_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {"model": model, "genre_encoder": genre_encoder, "feature_cols": FEATURE_COLS}, f
        )
    print(f"\nSaved -> {model_path}")

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    results = {
        "best_iteration": model.best_iteration,
        "feature_cols": FEATURE_COLS,
        "n_rows": {"train": len(train), "val": len(val), "test": len(test)},
        "val": {
            "baseline_rmse": round(baseline_rmse, 3),
            "baseline_mae": round(baseline_mae, 3),
            "lightgbm_rmse": round(model_rmse, 3),
            "lightgbm_mae": round(model_mae, 3),
            "rmse_improvement_pct": round(val_rmse_improvement, 1),
            "mae_improvement_pct": round(val_mae_improvement, 1),
        },
        "test": {
            "baseline_rmse": round(test_baseline_rmse, 3),
            "baseline_mae": round(test_baseline_mae, 3),
            "lightgbm_rmse": round(test_model_rmse, 3),
            "lightgbm_mae": round(test_model_mae, 3),
            "rmse_improvement_pct": round(test_rmse_improvement, 1),
            "mae_improvement_pct": round(test_mae_improvement, 1),
        },
    }
    results_path = MODELS_DIR / "forecasting_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {results_path}")

    # ------------------------------------------------------------------
    # 9. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Val   -- Baseline RMSE: {baseline_rmse:.3f}   LightGBM RMSE: {model_rmse:.3f}   Improvement: {val_rmse_improvement:+.1f}%")
    print(f"Test  -- Baseline RMSE: {test_baseline_rmse:.3f}   LightGBM RMSE: {test_model_rmse:.3f}   Improvement: {test_rmse_improvement:+.1f}%")


if __name__ == "__main__":
    main()
