"""
Phase 3: account-network graph features, built with networkx.

Money laundering often isn't visible in a single transaction -- it shows up
in the SHAPE of an account's connections: does it send to many different
people in a short time (fan-out), receive from many different people
(fan-in), or sit in an unusually central position in the money-flow network?
These features try to capture that.

Leakage rule: for each split (train/val/test), the account graph is built
ONLY from edges (transactions) up to the END of that split's own time range.
Test-set accounts can be influenced by train- and val-period transactions
(which genuinely happened earlier), but train/val features are NEVER
influenced by test-period transactions. This is a coarser rule than the
per-row "strictly before this exact transaction" rule used for the account
behaviour features in features.py -- within a single split, a transaction's
own edge can contribute to its own account's degree/PageRank for that same
split's snapshot. That's a deliberate, documented simplification (matching
how this phase is scoped in the project brief), not an oversight: what it
never does is let a LATER split's data leak into an EARLIER one.

Run from the project root, with the virtual environment active:
    python -m src.graph_features
"""

from pathlib import Path

import networkx as nx
import pandas as pd

from src.config import load_config
from src.features import build_features


def build_account_graph(edges: pd.DataFrame) -> nx.DiGraph:
    """
    One directed edge per (from_id, to_id) pair, weighted by the SUM of
    amount_usd across every transaction between that pair. Multiple
    transactions between the same two accounts collapse into one weighted
    edge rather than being kept as separate parallel edges -- simpler, and
    everything computed below (degree, PageRank) only needs this summary.
    """
    pair_totals = edges.groupby(["from_id", "to_id"], observed=True)["amount_usd"].sum().reset_index()
    return nx.from_pandas_edgelist(
        pair_totals, source="from_id", target="to_id", edge_attr="amount_usd", create_using=nx.DiGraph
    )


def compute_account_graph_features(G: nx.DiGraph, pagerank_alpha: float) -> pd.DataFrame:
    """
    One row per account. Each feature and why it relates to laundering:

    - in_degree / out_degree: how many DIFFERENT accounts sent to / received
      from this account (fan-in / fan-out). A normal personal account might
      have a handful of counterparties; a layering account moving money
      through many shells at once often has an unusually high degree.
    - total_in_amount / total_out_amount: total USD received / sent. An
      account passing through roughly what it takes in (in ~ out) is a
      classic "pass-through" / layering pattern, rather than genuinely
      spending or saving the money.
    - pagerank: a measure of how "central" an account is in the money-flow
      network, weighted by amount -- an account is scored higher if money
      flows through it from OTHER already-central accounts. Useful for
      spotting hub accounts that sit at the middle of a laundering network,
      not just ones with a high raw transaction count.
    """
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    total_in = dict(G.in_degree(weight="amount_usd"))
    total_out = dict(G.out_degree(weight="amount_usd"))
    pagerank = nx.pagerank(G, alpha=pagerank_alpha, weight="amount_usd")

    accounts = list(G.nodes())
    return pd.DataFrame({
        "account_id": accounts,
        "in_degree": [in_degree.get(a, 0) for a in accounts],
        "out_degree": [out_degree.get(a, 0) for a in accounts],
        "total_in_amount": [total_in.get(a, 0.0) for a in accounts],
        "total_out_amount": [total_out.get(a, 0.0) for a in accounts],
        "pagerank": [pagerank.get(a, 0.0) for a in accounts],
    })


def add_graph_features(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, pagerank_alpha: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute graph features per split, each from a graph built only on edges
    up to that split's own end -- see the module docstring for the exact rule.
    """
    graph_source_by_split = {
        "train": train,
        "val": pd.concat([train, val], ignore_index=True),
        "test": pd.concat([train, val, test], ignore_index=True),
    }
    splits = {"train": train, "val": val, "test": test}

    graph_feature_cols = ["in_degree", "out_degree", "total_in_amount", "total_out_amount", "pagerank"]
    fill_values = {col: 0 for col in graph_feature_cols}

    results = {}
    for name, df in splits.items():
        G = build_account_graph(graph_source_by_split[name])
        account_feats = compute_account_graph_features(G, pagerank_alpha)

        merged = df.merge(
            account_feats.add_prefix("from_").rename(columns={"from_account_id": "from_id"}),
            on="from_id", how="left",
        )
        merged = merged.merge(
            account_feats.add_prefix("to_").rename(columns={"to_account_id": "to_id"}),
            on="to_id", how="left",
        )
        from_cols = {f"from_{c}": v for c, v in fill_values.items()}
        to_cols = {f"to_{c}": v for c, v in fill_values.items()}
        merged = merged.fillna({**from_cols, **to_cols})
        results[name] = merged

    return results["train"], results["val"], results["test"]


def save_outputs(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, config: dict) -> None:
    features_dir = Path(config["paths"]["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(features_dir / "train_features.parquet", index=False)
    val.to_parquet(features_dir / "val_features.parquet", index=False)
    test.to_parquet(features_dir / "test_features.parquet", index=False)


def main() -> None:
    config = load_config()
    train, val, test = build_features(config)
    train, val, test = add_graph_features(train, val, test, config["features"]["pagerank_alpha"])
    save_outputs(train, val, test, config)
    print(f"train: {len(train):,} rows, {train.shape[1]} columns")
    print(f"val:   {len(val):,} rows, {val.shape[1]} columns")
    print(f"test:  {len(test):,} rows, {test.shape[1]} columns")


if __name__ == "__main__":
    main()
