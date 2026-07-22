"""Sample, split, and report on the engineered CineIQ feature tables.

Downstream of feature_engineering.py. Takes the full rec_features.csv and
rl_features.csv (one row per rating event, ~25M rows each) and reduces them
to a 5M-row sample each, then produces train/val/test splits for all three
components: a temporal split for the forecasting features (train on the
past, validate/test on more recent periods) and random 70/15/15 splits for
the recommendation and RL features. Every output file gets a row count,
date range (where applicable), and peak memory usage printed at the end.
"""

import threading
from pathlib import Path

import pandas as pd
import psutil

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES_DIR = DATA_DIR / "features"

RANDOM_STATE = 42
SAMPLE_SIZE = 5_000_000
CASUAL_MAX = 20
REGULAR_MAX = 100

REC_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "user_total_ratings": "int32",
    "user_avg_rating": "float32",
    "user_rating_std": "float32",
    "user_favorite_genre": "category",
    "movie_total_ratings": "int32",
    "movie_avg_rating": "float32",
    "movie_rating_std": "float32",
    "movie_genres": "category",
}

RL_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "year": "int16",
    "month": "int8",
    "reward": "int8",
    "user_segment": "category",
    "recent_movie_ids": "object",
}


class PeakMemoryTracker:
    """Samples this process's RSS on a background thread to report a peak."""

    def __init__(self, interval=0.05):
        self._process = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.peak_bytes = 0

    def _run(self):
        while not self._stop.is_set():
            rss = self._process.memory_info().rss
            self.peak_bytes = max(self.peak_bytes, rss)
            self._stop.wait(self._interval)

    def __enter__(self):
        self.peak_bytes = self._process.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()

    @property
    def peak_mb(self):
        return self.peak_bytes / 1e6


def segment_label(n):
    if n <= CASUAL_MAX:
        return "casual"
    elif n <= REGULAR_MAX:
        return "regular"
    return "power"


def year_month_range(df):
    periods = pd.PeriodIndex.from_fields(year=df["year"], month=df["month"], freq="M")
    return f"{periods.min()} to {periods.max()}"


def random_three_way_split(df, train_frac=0.70, val_frac=0.15):
    train = df.sample(frac=train_frac, random_state=RANDOM_STATE)
    remaining = df.drop(train.index)
    val = remaining.sample(frac=val_frac / (1 - train_frac), random_state=RANDOM_STATE)
    test = remaining.drop(val.index)
    return train, val, test


def main():
    results = []

    # ----------------------------------------------------------------
    # 2. Stratified sample of rec_features.csv, delete the original
    # ----------------------------------------------------------------
    print("=" * 70)
    print("2. STRATIFIED SAMPLE: rec_features.csv -> rec_features_sampled.csv")
    print("=" * 70)
    with PeakMemoryTracker() as tracker:
        rec = pd.read_csv(FEATURES_DIR / "rec_features.csv", dtype=REC_DTYPES)
        print(f"Loaded rec_features.csv: {rec.shape}")

        segment = rec["user_total_ratings"].apply(segment_label)
        frac = SAMPLE_SIZE / len(rec)
        rec_sampled = pd.concat(
            [group.sample(frac=frac, random_state=RANDOM_STATE) for _, group in rec.groupby(segment)]
        )
        if len(rec_sampled) > SAMPLE_SIZE:
            rec_sampled = rec_sampled.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
        rec_sampled = rec_sampled.sort_index()
        rec_sampled.to_csv(FEATURES_DIR / "rec_features_sampled.csv", index=False)
        del rec
    results.append(
        {"file": "rec_features_sampled.csv", "rows": len(rec_sampled),
         "date_range": "N/A (no date column)", "peak_mb": tracker.peak_mb}
    )
    print(f"Saved rec_features_sampled.csv: {rec_sampled.shape}  (peak memory {tracker.peak_mb:.1f} MB)")
    print(rec_sampled["user_total_ratings"].apply(segment_label).value_counts())

    (FEATURES_DIR / "rec_features.csv").unlink()
    print("Deleted original rec_features.csv")

    # ----------------------------------------------------------------
    # 3. Random sample of rl_features.csv, delete the original
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. RANDOM SAMPLE: rl_features.csv -> rl_features_sampled.csv")
    print("=" * 70)
    with PeakMemoryTracker() as tracker:
        rl = pd.read_csv(FEATURES_DIR / "rl_features.csv", dtype=RL_DTYPES)
        print(f"Loaded rl_features.csv: {rl.shape}")
        rl_sampled = rl.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
        rl_sampled.to_csv(FEATURES_DIR / "rl_features_sampled.csv", index=False)
        del rl
    results.append(
        {"file": "rl_features_sampled.csv", "rows": len(rl_sampled),
         "date_range": year_month_range(rl_sampled), "peak_mb": tracker.peak_mb}
    )
    print(f"Saved rl_features_sampled.csv: {rl_sampled.shape}  (peak memory {tracker.peak_mb:.1f} MB)")

    (FEATURES_DIR / "rl_features.csv").unlink()
    print("Deleted original rl_features.csv")

    # ----------------------------------------------------------------
    # 4. Temporal split of forecasting_features.csv
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. TEMPORAL SPLIT: forecasting_features.csv")
    print("=" * 70)
    with PeakMemoryTracker() as tracker:
        forecasting = pd.read_csv(FEATURES_DIR / "forecasting_features.csv")
        forecasting["period"] = pd.PeriodIndex.from_fields(
            year=forecasting["year"], month=forecasting["month"], freq="M"
        )
        periods = sorted(forecasting["period"].unique())
        n_periods = len(periods)
        n_test = max(1, round(n_periods * 0.10))
        n_val = max(1, round(n_periods * 0.10))

        test_periods = set(periods[-n_test:])
        val_periods = set(periods[-(n_test + n_val):-n_test])
        train_periods = set(periods[: -(n_test + n_val)])

        forecasting_train = forecasting[forecasting["period"].isin(train_periods)].drop(columns=["period"])
        forecasting_val = forecasting[forecasting["period"].isin(val_periods)].drop(columns=["period"])
        forecasting_test = forecasting[forecasting["period"].isin(test_periods)].drop(columns=["period"])

        forecasting_train.to_csv(FEATURES_DIR / "forecasting_train.csv", index=False)
        forecasting_val.to_csv(FEATURES_DIR / "forecasting_val.csv", index=False)
        forecasting_test.to_csv(FEATURES_DIR / "forecasting_test.csv", index=False)
    peak = tracker.peak_mb

    for name, df in [
        ("forecasting_train.csv", forecasting_train),
        ("forecasting_val.csv", forecasting_val),
        ("forecasting_test.csv", forecasting_test),
    ]:
        results.append({"file": name, "rows": len(df), "date_range": year_month_range(df), "peak_mb": peak})
        print(f"{name}: {df.shape}  periods {year_month_range(df)}")

    # ----------------------------------------------------------------
    # 5. Random 70/15/15 split of rec_features_sampled.csv
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. RANDOM SPLIT (70/15/15): rec_features_sampled.csv")
    print("=" * 70)
    with PeakMemoryTracker() as tracker:
        rec_sampled = pd.read_csv(FEATURES_DIR / "rec_features_sampled.csv", dtype=REC_DTYPES)
        rec_train, rec_val, rec_test = random_three_way_split(rec_sampled)
        rec_train.to_csv(FEATURES_DIR / "rec_train.csv", index=False)
        rec_val.to_csv(FEATURES_DIR / "rec_val.csv", index=False)
        rec_test.to_csv(FEATURES_DIR / "rec_test.csv", index=False)
    peak = tracker.peak_mb

    for name, df in [("rec_train.csv", rec_train), ("rec_val.csv", rec_val), ("rec_test.csv", rec_test)]:
        results.append({"file": name, "rows": len(df), "date_range": "N/A (no date column)", "peak_mb": peak})
        print(f"{name}: {df.shape}")

    # ----------------------------------------------------------------
    # 6. Random 70/15/15 split of rl_features_sampled.csv
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. RANDOM SPLIT (70/15/15): rl_features_sampled.csv")
    print("=" * 70)
    with PeakMemoryTracker() as tracker:
        rl_sampled_df = pd.read_csv(FEATURES_DIR / "rl_features_sampled.csv", dtype=RL_DTYPES)
        rl_train, rl_val, rl_test = random_three_way_split(rl_sampled_df)
        rl_train.to_csv(FEATURES_DIR / "rl_train.csv", index=False)
        rl_val.to_csv(FEATURES_DIR / "rl_val.csv", index=False)
        rl_test.to_csv(FEATURES_DIR / "rl_test.csv", index=False)
    peak = tracker.peak_mb

    for name, df in [("rl_train.csv", rl_train), ("rl_val.csv", rl_val), ("rl_test.csv", rl_test)]:
        results.append({"file": name, "rows": len(df), "date_range": year_month_range(df), "peak_mb": peak})
        print(f"{name}: {df.shape}  periods {year_month_range(df)}")

    # ----------------------------------------------------------------
    # 7. Final report
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("7. FINAL REPORT - ALL OUTPUT FILES")
    print("=" * 70)
    report_df = pd.DataFrame(results)
    report_df["rows"] = report_df["rows"].map(lambda n: f"{n:,}")
    report_df["peak_mb"] = report_df["peak_mb"].round(1)
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
