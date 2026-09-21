"""
Phase 3: transaction and account-behaviour features.

Every account-behaviour feature is built so it can only see a transaction's
STRICTLY PAST history -- never the transaction itself, never anything at the
same timestamp (ties are common: this dataset only has minute-level
precision), and never anything later. See the "no future data" check at the
bottom of notebooks/03_feature_engineering.ipynb for a test of this.

Run from the project root, with the virtual environment active:
    python -m src.features
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config


def time_based_split(df: pd.DataFrame, split_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train/val/test by ROW POSITION after sorting by time -- never
    randomly. Train gets the earliest rows, then val, then test gets the most
    recent rows. This mirrors how the model will actually be used: trained on
    the past, evaluated on data that comes strictly after it.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * split_cfg["train_frac"])
    val_end = train_end + int(n * split_cfg["val_frac"])

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)
    return train, val, test


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features that only use a transaction's own fields -- no history involved,
    so there's no leakage risk here.
    """
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount_usd"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_cross_currency"] = df["payment_currency"] != df["receiving_currency"]
    df["is_cross_bank"] = df["from_bank"] != df["to_bank"]
    return df


def _past_only_rolling(df: pd.DataFrame, id_col: str, value_col: str, window: str, agg: str) -> pd.Series:
    """
    For each row, aggregate `value_col` over the SAME id_col group's rows
    strictly before this row's timestamp, within the trailing `window`.

    closed="left" is what makes this leakage-safe: a time-based rolling
    window normally includes the current row (the window's right edge). With
    closed="left" the right edge is excluded, so the current row -- and any
    other row sharing its exact timestamp -- is left out of its own window.

    min_periods=0 makes "count"/"sum" return 0 (not NaN) when an account has
    no prior history yet -- pandas' default min_periods=1 would otherwise
    treat a genuinely empty window as missing data. "mean"/"max" still come
    out NaN with zero prior rows, which is correct: there's no sensible
    average or max of nothing.
    """
    ordered = df.sort_values([id_col, "timestamp"])
    rolled = (
        ordered.set_index("timestamp")
        .groupby(id_col)[value_col]
        .rolling(window, closed="left", min_periods=0)
        .agg(agg)
    )
    # rolled is indexed by (id_col, timestamp) in the same row order as `ordered`,
    # so it can be reattached positionally, then reordered back to df's original rows.
    result = pd.Series(rolled.values, index=ordered.index)
    return result.reindex(df.index)


def add_account_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Account-behaviour features, computed from each account's history strictly
    before the current transaction only. Must be called on the FULL
    chronologically-ordered dataset (not per-split) -- see the module
    docstring and the Phase 3 write-up in the notebook for why that's
    actually the correct thing to do, not a leak.
    """
    df = df.copy()

    # --- Sent-side features (grouped by the account that's sending) ---
    df["sent_count_1d"] = _past_only_rolling(df, "from_id", "amount_usd", "1D", "count")
    df["sent_count_7d"] = _past_only_rolling(df, "from_id", "amount_usd", "7D", "count")
    df["sent_avg_amount_7d"] = _past_only_rolling(df, "from_id", "amount_usd", "7D", "mean")
    df["sent_max_amount_7d"] = _past_only_rolling(df, "from_id", "amount_usd", "7D", "max")

    # Unique counterparties sent to in the last 7 days. Rolling only works on
    # numbers, so to_id is turned into integer codes first, then the window
    # counts distinct codes.
    df["_to_id_code"] = df["to_id"].astype("category").cat.codes
    ordered = df.sort_values(["from_id", "timestamp"])
    unique_recv_7d = (
        ordered.set_index("timestamp")
        .groupby("from_id")["_to_id_code"]
        .rolling("7D", closed="left", min_periods=0)
        .apply(lambda arr: len(set(arr)), raw=True)
    )
    df["unique_counterparties_7d"] = pd.Series(unique_recv_7d.values, index=ordered.index).reindex(df.index)
    df = df.drop(columns="_to_id_code")

    # "Usual" amount for the account = mean of ALL its strictly-prior sent
    # transactions (a very wide window stands in for "all of history so far").
    df["sent_avg_amount_alltime"] = _past_only_rolling(df, "from_id", "amount_usd", "3650D", "mean")
    df["amount_vs_usual_ratio"] = df["amount_usd"] / df["sent_avg_amount_alltime"]

    # Time since this account's previous transaction (as sender). Rows that
    # share an exact timestamp with the row before them (same account) get
    # 0 here -- not a leak, just an honest "we can't tell which came first."
    ordered = df.sort_values(["from_id", "timestamp"])
    prev_ts = ordered.groupby("from_id")["timestamp"].shift(1)
    minutes_since = (ordered["timestamp"] - prev_ts).dt.total_seconds() / 60
    df["minutes_since_prev_txn"] = pd.Series(minutes_since.values, index=ordered.index).reindex(df.index)

    # --- Received-side counts (grouped by the account that's receiving) ---
    df["received_count_1d"] = _past_only_rolling(df, "to_id", "amount_usd", "1D", "count")
    df["received_count_7d"] = _past_only_rolling(df, "to_id", "amount_usd", "7D", "count")

    return df


def build_features(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cleaned data, add all features, then split into train/val/test."""
    df = pd.read_parquet(config["paths"]["clean_file"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = add_transaction_features(df)
    df = add_account_behavior_features(df)

    train, val, test = time_based_split(df, config["split"])
    return train, val, test


def save_outputs(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, config: dict) -> None:
    features_dir = Path(config["paths"]["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(features_dir / "train_features.parquet", index=False)
    val.to_parquet(features_dir / "val_features.parquet", index=False)
    test.to_parquet(features_dir / "test_features.parquet", index=False)


def main() -> None:
    config = load_config()
    train, val, test = build_features(config)
    save_outputs(train, val, test, config)
    print(f"train: {len(train):,} rows ({train['timestamp'].min()} to {train['timestamp'].max()})")
    print(f"val:   {len(val):,} rows ({val['timestamp'].min()} to {val['timestamp'].max()})")
    print(f"test:  {len(test):,} rows ({test['timestamp'].min()} to {test['timestamp'].max()})")


if __name__ == "__main__":
    main()
