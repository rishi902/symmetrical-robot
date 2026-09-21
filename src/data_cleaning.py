"""
Phase 1: load the raw AML transactions CSV, clean it, and save a cleaned
parquet file for the rest of the pipeline to use.

Run this from the project root, with the virtual environment active:
    python -m src.data_cleaning

It reads data/raw/HI-Small_Trans.csv (see config.yaml for the exact path),
writes the cleaned data to data/processed/transactions_clean.parquet, and
writes a plain-language report of every cleaning decision to
results/metrics/cleaning_report.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config

# The raw Kaggle columns use spaces and mixed case. We rename them to
# consistent snake_case names so the rest of the code is easier to read.
COLUMN_RENAME_MAP = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account.1": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}

# This is the literal string used for US Dollars in this dataset's currency
# columns (confirmed by inspecting the raw data), not just "USD".
USD_LABEL = "US Dollar"


def load_raw_data(raw_file: Path) -> pd.DataFrame:
    """Load the raw CSV and rename columns to snake_case."""
    df = pd.read_csv(raw_file)
    df = df.rename(columns=COLUMN_RENAME_MAP)
    return df


def report_raw_summary(df: pd.DataFrame) -> str:
    """Build a short text summary of the raw data: shape, dtypes, memory, missing values, duplicates."""
    lines = ["## Raw data summary", ""]
    lines.append(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    lines.append(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    lines.append("")
    lines.append("Dtypes:")
    for col, dtype in df.dtypes.items():
        lines.append(f"  {col}: {dtype}")
    lines.append("")

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        lines.append("Missing values: none")
    else:
        lines.append("Missing values:")
        for col, n in missing.items():
            lines.append(f"  {col}: {n:,} ({n / len(df):.2%})")
    lines.append("")

    n_dupes = df.duplicated().sum()
    lines.append(f"Exact duplicate rows: {n_dupes:,} ({n_dupes / len(df):.2%})")
    return "\n".join(lines)


def make_unique_account_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Account numbers in this dataset are only guaranteed unique WITHIN a bank,
    not across banks. Two different banks could both have an account
    "8000EBD30". We build a combined id (bank + account) so accounts are
    globally unique, and use that everywhere downstream instead of the raw
    account number.
    """
    df = df.copy()
    df["from_id"] = df["from_bank"].astype(str) + "_" + df["from_account"].astype(str)
    df["to_id"] = df["to_bank"].astype(str) + "_" + df["to_account"].astype(str)
    return df


def sample_by_accounts(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """
    Sample a random subset of accounts, then keep every transaction that
    touches at least one of them (as sender OR receiver) -- not just
    transactions between two sampled accounts. This keeps each sampled
    account's real neighbourhood intact (its actual counterparties, in-degree,
    out-degree), instead of artificially cutting it off from the rest of the
    network, which would understate its connectivity for Phase 3's graph
    features.
    """
    all_accounts = pd.unique(pd.concat([df["from_id"], df["to_id"]]))
    rng = np.random.default_rng(seed)
    n_keep = int(len(all_accounts) * frac)
    sampled_accounts = set(rng.choice(all_accounts, size=n_keep, replace=False))

    mask = df["from_id"].isin(sampled_accounts) | df["to_id"].isin(sampled_accounts)
    return df[mask].reset_index(drop=True)


def check_account_id_collision(df: pd.DataFrame) -> str:
    """
    Check whether account numbers actually do collide across banks, to see
    whether from_id/to_id were necessary or just a precaution.
    """
    accounts_per_bank_count = df.groupby("from_account")["from_bank"].nunique()
    n_colliding = int((accounts_per_bank_count > 1).sum())
    total_accounts = len(accounts_per_bank_count)
    return (
        "## Account id check\n\n"
        f"{n_colliding:,} of {total_accounts:,} account numbers "
        f"({n_colliding / total_accounts:.2%}) appear under more than one bank. "
        + (
            "This confirms account numbers are NOT globally unique, so combining "
            "bank + account into from_id/to_id was necessary."
            if n_colliding > 0
            else "No collisions were found in this sample, but we keep from_id/to_id "
            "anyway since the dataset documentation says collisions are possible."
        )
    )


def infer_exchange_rates(df: pd.DataFrame) -> tuple[dict, str]:
    """
    Estimate a currency -> USD conversion rate for every currency in the
    data, using rows where one side of the transfer is already in USD.

    Why this works: amount_paid (in payment_currency) and amount_received
    (in receiving_currency) represent the same value on both sides of one
    transfer. So when one side is USD, the ratio between the two amounts is
    an estimate of the other currency's exchange rate to USD. We collect
    every such ratio for a currency and take the median, which is robust to
    a handful of noisy or unusual transactions.
    """
    rates = {USD_LABEL: 1.0}
    lines = ["## Inferred exchange rates (to USD)", ""]

    all_currencies = pd.unique(pd.concat([df["payment_currency"], df["receiving_currency"]]))
    for currency in sorted(all_currencies):
        if currency == USD_LABEL:
            continue

        # Case 1: paid in USD, received in this currency.
        mask_a = (df["payment_currency"] == USD_LABEL) & (df["receiving_currency"] == currency)
        ratios_a = df.loc[mask_a, "amount_paid"] / df.loc[mask_a, "amount_received"]

        # Case 2: received in USD, paid in this currency.
        mask_b = (df["receiving_currency"] == USD_LABEL) & (df["payment_currency"] == currency)
        ratios_b = df.loc[mask_b, "amount_received"] / df.loc[mask_b, "amount_paid"]

        ratios = pd.concat([ratios_a, ratios_b]).replace([np.inf, -np.inf], np.nan).dropna()

        if len(ratios) == 0:
            # No rows link this currency to USD directly in this sample.
            # Default to 1.0 (no conversion) rather than guessing, and flag
            # it clearly so it's not mistaken for a real rate.
            rates[currency] = 1.0
            lines.append(f"  {currency}: no USD-linked rows found, defaulted to 1.0 (FLAGGED, check this)")
            continue

        rate = float(ratios.median())
        rates[currency] = rate
        # A rate estimated from only a handful of rows is much less trustworthy
        # than one backed by hundreds. Flag it instead of presenting every
        # rate with the same confidence.
        low_confidence_note = " (LOW CONFIDENCE: fewer than 30 supporting rows)" if len(ratios) < 30 else ""
        lines.append(
            f"  {currency}: rate={rate:.6f}, based on {len(ratios):,} rows, "
            f"std={ratios.std():.6f}{low_confidence_note}"
        )

    return rates, "\n".join(lines)


def clean_data(config: dict) -> tuple[pd.DataFrame, str]:
    """Run the full Phase 1 cleaning pipeline and return (cleaned_df, report_text)."""
    paths = config["paths"]
    cleaning_cfg = config["cleaning"]
    runtime_cfg = config["runtime"]

    report = ["# Data Cleaning Report", ""]

    df = load_raw_data(Path(paths["raw_file"]))
    report.append(report_raw_summary(df))
    report.append("")

    # Parse timestamps and sort chronologically. Everything downstream
    # (sampling, splitting, rolling features) depends on this order.
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Account ids are built here, BEFORE sampling, because account-based
    # sampling (below) needs them to decide which rows to keep.
    df = make_unique_account_ids(df)
    report.append(check_account_id_collision(df))
    report.append("")

    # Develop on a smaller slice while the pipeline is being built (see
    # runtime.sample_frac / sample_mode in config.yaml).
    sample_frac = runtime_cfg["sample_frac"]
    if sample_frac < 1.0:
        sample_mode = runtime_cfg["sample_mode"]
        if sample_mode == "accounts":
            before = len(df)
            df = sample_by_accounts(df, sample_frac, seed=config["project"]["random_seed"])
            report.append(
                f"Sampled to {sample_frac:.0%} of accounts, keeping every transaction that touches "
                f"one of them ({before:,} -> {len(df):,} rows, sample_mode=accounts). This is done "
                "instead of taking the first N% of rows because this dataset is extremely "
                "front-loaded in time (99.98% of transactions happen in the first 10 of 18 days -- "
                "see the EDA notebook), so a row-count sample would only cover a few minutes and "
                "give every account a near-empty history. Sampling by account instead means every "
                "included account keeps its FULL history across the whole time range, which the "
                "time-window features in Phase 3 need."
            )
        elif sample_mode == "head":
            # Take the EARLIEST rows, not a random sample, so we don't fake up
            # relationships between rows that were never actually adjacent in time.
            # Kept as an option, but see the note above on why "accounts" is the default.
            n_keep = int(len(df) * sample_frac)
            df = df.head(n_keep).reset_index(drop=True)
            report.append(f"Sampled to {sample_frac:.0%} of rows ({n_keep:,} rows), taken from the start (sample_mode=head).")
        else:
            raise NotImplementedError(f"sample_mode={sample_mode!r} is not implemented")
        report.append("")

    # --- Exact duplicates ---
    if cleaning_cfg["drop_exact_duplicates"]:
        before = len(df)
        n_dupes = int(df.duplicated().sum())
        df = df.drop_duplicates().reset_index(drop=True)
        report.append(
            "## Exact duplicates\n\n"
            f"Dropped {n_dupes:,} exact duplicate rows ({before:,} -> {len(df):,}). "
            "These are data entry repeats, not genuine repeated transfers."
        )
        report.append("")

    # --- Invalid amounts (zero or negative) ---
    before = len(df)
    amount_min = cleaning_cfg["amount_min"]
    invalid_mask = (df["amount_paid"] <= amount_min) | (df["amount_received"] <= amount_min)
    n_invalid = int(invalid_mask.sum())
    df = df[~invalid_mask].reset_index(drop=True)
    report.append(
        "## Invalid amounts\n\n"
        f"Dropped {n_invalid:,} rows with amount_paid or amount_received <= {amount_min} "
        f"({before:,} -> {len(df):,}). A transaction of zero or negative value can't be a "
        "real transfer, so these are treated as broken rows rather than genuine data."
    )
    report.append("")

    # --- Self-transfers: kept, just flagged ---
    if not cleaning_cfg["drop_self_transfers"]:
        df["is_self_transfer"] = df["from_id"] == df["to_id"]
        n_self = int(df["is_self_transfer"].sum())

        # Break the self-transfer rate down by payment_format. A high overall
        # rate is only trustworthy if we can see WHICH format is driving it,
        # rather than just accepting the headline number.
        format_breakdown = df.groupby("payment_format", observed=True)["is_self_transfer"].mean().sort_values(ascending=False)
        format_lines = "\n".join(f"  {fmt}: {rate:.1%} self-transfers" for fmt, rate in format_breakdown.items())

        report.append(
            "## Self-transfers\n\n"
            f"{n_self:,} rows ({n_self / len(df):.2%}) are self-transfers (same account on both sides). "
            "Kept (not dropped) and flagged with a new is_self_transfer column, since this is a "
            "real account behaviour pattern, not a data error.\n\n"
            f"Self-transfer rate by payment_format:\n{format_lines}"
        )
        report.append("")

    # --- Extreme outliers: kept deliberately, just reported ---
    amount_usd_preview = df["amount_paid"].where(df["payment_currency"] == USD_LABEL, np.nan)
    report.append(
        "## Extreme outliers\n\n"
        "Outlier amounts are NOT removed. A very large or unusual transaction is exactly the kind "
        "of signal we're trying to detect, so deleting it could throw away real laundering cases. "
        "Skew is handled later with a log-amount feature (Phase 3) instead of deleting rows here.\n\n"
        f"amount_paid (USD-labelled rows only, for a quick look): "
        f"min={amount_usd_preview.min():.2f}, "
        f"median={amount_usd_preview.median():.2f}, "
        f"max={amount_usd_preview.max():.2f}"
    )
    report.append("")

    # --- Currency conversion to USD ---
    if cleaning_cfg["infer_exchange_rates"]:
        rates, rate_report = infer_exchange_rates(df)
        df["amount_paid_usd"] = df["amount_paid"] * df["payment_currency"].map(rates)
        df["amount_received_usd"] = df["amount_received"] * df["receiving_currency"].map(rates)
        # amount_usd is the single canonical transaction value used from here on:
        # the paid amount (sender's side), converted to USD.
        df["amount_usd"] = df["amount_paid_usd"]

        # Sanity check: paid and received should convert to roughly the same
        # USD value, since they're two sides of the same transfer.
        pct_diff = ((df["amount_paid_usd"] - df["amount_received_usd"]).abs() / df["amount_paid_usd"]).median()
        report.append(rate_report)
        report.append("")
        report.append(
            f"Sanity check: median relative difference between amount_paid_usd and "
            f"amount_received_usd is {pct_diff:.2%} (small values here mean the inferred rates are consistent)."
        )
        report.append("")
    else:
        raise NotImplementedError("infer_exchange_rates=false is not implemented; no fixed rate table is provided")

    # --- Categorical dtypes (memory optimisation, no value changes) ---
    before_mb = df.memory_usage(deep=True).sum() / 1e6
    cat_cols = ["from_bank", "to_bank", "receiving_currency", "payment_currency", "payment_format"]
    for col in cat_cols:
        df[col] = df[col].astype("category")
    df["is_laundering"] = df["is_laundering"].astype("int8")
    after_mb = df.memory_usage(deep=True).sum() / 1e6
    report.append(
        "## Dtype optimisation\n\n"
        f"Converted {', '.join(cat_cols)} to category dtype, and is_laundering to int8. "
        f"Memory usage: {before_mb:.1f} MB -> {after_mb:.1f} MB."
    )
    report.append("")

    report.append(f"## Final shape\n\n{df.shape[0]:,} rows x {df.shape[1]} columns")

    return df, "\n".join(report)


def save_outputs(df: pd.DataFrame, report: str, config: dict) -> None:
    """Save the cleaned parquet file and the cleaning report to the paths in config.yaml."""
    paths = config["paths"]

    processed_dir = Path(paths["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths["clean_file"], index=False)

    metrics_dir = Path(paths["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_path = metrics_dir / "cleaning_report.md"
    report_path.write_text(report)


def main() -> None:
    config = load_config()
    df, report = clean_data(config)
    save_outputs(df, report, config)
    print(report)
    print(f"\nSaved cleaned data to {config['paths']['clean_file']}")
    print(f"Saved cleaning report to {Path(config['paths']['metrics_dir']) / 'cleaning_report.md'}")


if __name__ == "__main__":
    main()
