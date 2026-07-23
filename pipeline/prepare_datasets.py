"""Split the engineered CineIQ feature tables into train/val/test sets.

- forecasting_features.csv: temporal split by week period.
- rl_features.parquet: temporal split by timestamp (global, row-level) for
  the RL agent's interaction stream.
- Recommendation split: leave-last-5-out per user on rl_features.parquet --
  each user's last 5 ratings become test, their prior 5 become validation,
  and everything before that is train. Users need at least 11 ratings to
  have history left over for all three splits.
- user_features.parquet / movie_features.parquet: random 80/10/10 split.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES_DIR = DATA_DIR / "features"

RANDOM_STATE = 42
LEAVE_OUT_TEST = 5
LEAVE_OUT_VAL = 5
MIN_RATINGS_FOR_REC_SPLIT = LEAVE_OUT_TEST + LEAVE_OUT_VAL + 1  # 11: needs >=1 rating left for train


def week_range(df):
    dates = pd.to_datetime(df["week_start"])
    return f"{dates.min().date()} to {dates.max().date()}"


def timestamp_range(df):
    dates = pd.to_datetime(df["timestamp"], unit="s")
    return f"{dates.min().date()} to {dates.max().date()}"


def random_three_way_split(df, train_frac=0.80, val_frac=0.10, random_state=RANDOM_STATE):
    train = df.sample(frac=train_frac, random_state=random_state)
    remaining = df.drop(train.index)
    val = remaining.sample(frac=val_frac / (1 - train_frac), random_state=random_state)
    test = remaining.drop(val.index)
    return train, val, test


# ------------------------------------------------------------------
# 1. forecasting_features.csv -- temporal split by week period
# ------------------------------------------------------------------
def split_forecasting():
    df = pd.read_csv(FEATURES_DIR / "forecasting_features.csv", parse_dates=["week_start"])
    df["period"] = df["week_start"].dt.to_period("W")
    df["week_start"] = df["week_start"].dt.date  # keep output as plain YYYY-MM-DD, matching the source file

    periods = sorted(df["period"].unique())
    n_periods = len(periods)
    n_test = max(1, round(n_periods * 0.10))
    n_val = max(1, round(n_periods * 0.10))

    test_periods = set(periods[-n_test:])
    val_periods = set(periods[-(n_test + n_val):-n_test])
    train_periods = set(periods[: -(n_test + n_val)])

    train = df[df["period"].isin(train_periods)].drop(columns=["period"])
    val = df[df["period"].isin(val_periods)].drop(columns=["period"])
    test = df[df["period"].isin(test_periods)].drop(columns=["period"])

    train.to_csv(FEATURES_DIR / "forecasting_train.csv", index=False)
    val.to_csv(FEATURES_DIR / "forecasting_val.csv", index=False)
    test.to_csv(FEATURES_DIR / "forecasting_test.csv", index=False)

    return train, val, test


# ------------------------------------------------------------------
# 2. rl_features.parquet -- temporal split by timestamp (row-level, global)
# ------------------------------------------------------------------
def split_rl_temporal(rl):
    df = rl.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    n_test = round(n * 0.10)
    n_val = round(n * 0.10)

    test = df.iloc[n - n_test:]
    val = df.iloc[n - n_test - n_val: n - n_test]
    train = df.iloc[: n - n_test - n_val]

    train.to_parquet(FEATURES_DIR / "rl_train.parquet", index=False)
    val.to_parquet(FEATURES_DIR / "rl_val.parquet", index=False)
    test.to_parquet(FEATURES_DIR / "rl_test.parquet", index=False)

    return train, val, test


# ------------------------------------------------------------------
# 3. Recommendation split -- leave-last-5-out per user on rl_features.parquet
# ------------------------------------------------------------------
def split_rec_leave_last_n_out(rl):
    user_counts = rl.groupby("userId").size()
    eligible_users = user_counts[user_counts >= MIN_RATINGS_FOR_REC_SPLIT].index
    n_excluded = user_counts.shape[0] - len(eligible_users)
    print(
        f"\nUsers eligible for leave-last-5-out (>= {MIN_RATINGS_FOR_REC_SPLIT} ratings): "
        f"{len(eligible_users):,} / {user_counts.shape[0]:,} ({n_excluded:,} excluded)"
    )

    eligible = rl[rl["userId"].isin(eligible_users)].copy()
    eligible = eligible.sort_values(["userId", "timestamp"], ascending=[True, False])
    eligible["rank_desc"] = eligible.groupby("userId").cumcount()  # 0 = most recent rating

    test = eligible[eligible["rank_desc"] < LEAVE_OUT_TEST].drop(columns=["rank_desc"])
    val = eligible[
        (eligible["rank_desc"] >= LEAVE_OUT_TEST) & (eligible["rank_desc"] < LEAVE_OUT_TEST + LEAVE_OUT_VAL)
    ].drop(columns=["rank_desc"])
    train = eligible[eligible["rank_desc"] >= LEAVE_OUT_TEST + LEAVE_OUT_VAL].drop(columns=["rank_desc"])

    train.to_parquet(FEATURES_DIR / "rec_train.parquet", index=False)
    val.to_parquet(FEATURES_DIR / "rec_val.parquet", index=False)
    test.to_parquet(FEATURES_DIR / "rec_test.parquet", index=False)

    return train, val, test


# ------------------------------------------------------------------
# 4. user_features.parquet / movie_features.parquet -- random 80/10/10 split
# ------------------------------------------------------------------
def split_random(path, out_prefix):
    df = pd.read_parquet(path)
    train, val, test = random_three_way_split(df)
    train.to_parquet(FEATURES_DIR / f"{out_prefix}_train.parquet", index=False)
    val.to_parquet(FEATURES_DIR / f"{out_prefix}_val.parquet", index=False)
    test.to_parquet(FEATURES_DIR / f"{out_prefix}_test.parquet", index=False)
    return train, val, test


def main():
    print("=" * 70)
    print("1. TEMPORAL SPLIT (by week period): forecasting_features.csv")
    print("=" * 70)
    f_train, f_val, f_test = split_forecasting()
    for name, df in [
        ("forecasting_train.csv", f_train),
        ("forecasting_val.csv", f_val),
        ("forecasting_test.csv", f_test),
    ]:
        print(f"{name}: {df.shape}  weeks {week_range(df)}")

    print("\nLoading rl_features.parquet...")
    rl = pd.read_parquet(FEATURES_DIR / "rl_features.parquet")
    print(f"rl_features.parquet: {rl.shape}")

    print("\n" + "=" * 70)
    print("2. TEMPORAL SPLIT (by timestamp): rl_features.parquet -> rl_train/val/test")
    print("=" * 70)
    rl_train, rl_val, rl_test = split_rl_temporal(rl)
    for name, df in [("rl_train.parquet", rl_train), ("rl_val.parquet", rl_val), ("rl_test.parquet", rl_test)]:
        print(f"{name}: {df.shape}  dates {timestamp_range(df)}")

    print("\n" + "=" * 70)
    print("3. LEAVE-LAST-5-OUT SPLIT: rl_features.parquet -> rec_train/val/test")
    print("=" * 70)
    rec_train, rec_val, rec_test = split_rec_leave_last_n_out(rl)
    for name, df in [("rec_train.parquet", rec_train), ("rec_val.parquet", rec_val), ("rec_test.parquet", rec_test)]:
        print(f"{name}: {df.shape}  dates {timestamp_range(df)}")

    print("\n" + "=" * 70)
    print("4. RANDOM SPLIT (80/10/10): user_features.parquet -> user_train/val/test")
    print("=" * 70)
    u_train, u_val, u_test = split_random(FEATURES_DIR / "user_features.parquet", "user")
    for name, df in [("user_train.parquet", u_train), ("user_val.parquet", u_val), ("user_test.parquet", u_test)]:
        print(f"{name}: {df.shape}")

    print("\n" + "=" * 70)
    print("4. RANDOM SPLIT (80/10/10): movie_features.parquet -> movie_train/val/test")
    print("=" * 70)
    m_train, m_val, m_test = split_random(FEATURES_DIR / "movie_features.parquet", "movie")
    for name, df in [("movie_train.parquet", m_train), ("movie_val.parquet", m_val), ("movie_test.parquet", m_test)]:
        print(f"{name}: {df.shape}")

    print("\n" + "=" * 70)
    print("5. FINAL REPORT - ALL OUTPUT FILES")
    print("=" * 70)
    rows = [
        ("forecasting_train.csv", f_train, week_range(f_train)),
        ("forecasting_val.csv", f_val, week_range(f_val)),
        ("forecasting_test.csv", f_test, week_range(f_test)),
        ("rl_train.parquet", rl_train, timestamp_range(rl_train)),
        ("rl_val.parquet", rl_val, timestamp_range(rl_val)),
        ("rl_test.parquet", rl_test, timestamp_range(rl_test)),
        ("rec_train.parquet", rec_train, timestamp_range(rec_train)),
        ("rec_val.parquet", rec_val, timestamp_range(rec_val)),
        ("rec_test.parquet", rec_test, timestamp_range(rec_test)),
        ("user_train.parquet", u_train, "N/A (no date column)"),
        ("user_val.parquet", u_val, "N/A (no date column)"),
        ("user_test.parquet", u_test, "N/A (no date column)"),
        ("movie_train.parquet", m_train, "N/A (no date column)"),
        ("movie_val.parquet", m_val, "N/A (no date column)"),
        ("movie_test.parquet", m_test, "N/A (no date column)"),
    ]
    report_df = pd.DataFrame(
        [{"file": name, "rows": f"{len(df):,}", "date_range": dr} for name, df, dr in rows]
    )
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
