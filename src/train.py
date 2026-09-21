"""
Phase 4: train a baseline XGBoost model and a logistic regression for
comparison. Both use genuinely DEFAULT, unweighted settings -- no
scale_pos_weight, no class_weight="balanced". That's deliberate: this is
the honest "before" picture. Comparing imbalance-handling methods is
Phase 5's job (see config.yaml: model.imbalance_method), not this one.

Only train and validation are used here. Test is never loaded in this
file -- it stays untouched until Phase 5 evaluates the final tuned model
on it exactly once.

Run from the project root, with the virtual environment active:
    python -m src.train
"""

from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_config
from src.evaluate import evaluate_model, save_metrics

TARGET_COL = "is_laundering"

NUMERIC_FEATURES = [
    "amount_usd", "log_amount", "hour", "day_of_week",
    "sent_count_1d", "sent_count_7d", "sent_avg_amount_7d", "sent_max_amount_7d",
    "unique_counterparties_7d", "sent_avg_amount_alltime", "amount_vs_usual_ratio",
    "minutes_since_prev_txn", "received_count_1d", "received_count_7d",
    "from_in_degree", "from_out_degree", "from_total_in_amount", "from_total_out_amount", "from_pagerank",
    "to_in_degree", "to_out_degree", "to_total_in_amount", "to_total_out_amount", "to_pagerank",
]
BOOLEAN_FEATURES = ["is_cross_currency", "is_cross_bank", "is_self_transfer"]
CATEGORICAL_FEATURES = ["payment_format", "payment_currency"]
FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


def load_feature_tables(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and val feature tables. Test is intentionally not loaded here."""
    features_dir = Path(config["paths"]["features_dir"])
    train_df = pd.read_parquet(features_dir / "train_features.parquet")
    val_df = pd.read_parquet(features_dir / "val_features.parquet")
    return train_df, val_df


def prepare_xgb_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    XGBoost can use category-dtype columns and missing values natively --
    many account-behaviour features are NaN for an account's very first
    transaction (see Phase 3), and XGBoost handles that directly rather
    than needing imputation.
    """
    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    for col in BOOLEAN_FEATURES:
        X[col] = X[col].astype(int)
    return X


def train_xgboost(train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict) -> xgb.XGBClassifier:
    X_train = prepare_xgb_data(train_df)
    y_train = train_df[TARGET_COL]
    X_val = prepare_xgb_data(val_df)
    y_val = val_df[TARGET_COL]

    params = dict(config["model"]["baseline_params"])
    model = xgb.XGBClassifier(
        **params,
        enable_categorical=True,
        tree_method="hist",
        eval_metric="aucpr",  # matches PR-AUC, our primary metric
        early_stopping_rounds=config["model"]["early_stopping_rounds"],
        random_state=config["project"]["random_seed"],
    )
    # eval_set is used only to decide WHEN to stop adding trees (early stopping),
    # not to fit any parameters -- but it does mean validation metrics are
    # very slightly optimistic, since the stopping point was chosen using
    # validation performance. This is standard practice, and is exactly why
    # the test set stays untouched until Phase 5's final, one-time check.
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def build_logreg_preprocessor() -> ColumnTransformer:
    """
    Unlike XGBoost, logistic regression can't handle missing values or raw
    category columns, and it's sensitive to feature scale (tree models
    aren't). So this imputes missing numeric values with the median,
    one-hot encodes the categoricals, and standardises everything.
    """
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("boolean", SimpleImputer(strategy="constant", fill_value=0), BOOLEAN_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def prepare_logreg_input(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean columns need to be cast to int -- sklearn's SimpleImputer doesn't accept bool dtype directly."""
    X = df[FEATURE_COLUMNS].copy()
    for col in BOOLEAN_FEATURES:
        X[col] = X[col].astype(int)
    return X


def train_logistic_regression(train_df: pd.DataFrame, config: dict) -> tuple[LogisticRegression, ColumnTransformer]:
    preprocessor = build_logreg_preprocessor()
    X_train = preprocessor.fit_transform(prepare_logreg_input(train_df))
    y_train = train_df[TARGET_COL]

    model = LogisticRegression(max_iter=1000, random_state=config["project"]["random_seed"])
    model.fit(X_train, y_train)
    return model, preprocessor


def main() -> None:
    config = load_config()
    train_df, val_df = load_feature_tables(config)
    y_val = val_df[TARGET_COL]

    print(f"train: {len(train_df):,} rows ({train_df[TARGET_COL].mean():.4%} laundering)")
    print(f"val:   {len(val_df):,} rows ({y_val.mean():.4%} laundering)")
    print()

    print("Training XGBoost (default settings)...")
    xgb_model = train_xgboost(train_df, val_df, config)
    xgb_scores = xgb_model.predict_proba(prepare_xgb_data(val_df))[:, 1]
    xgb_metrics = evaluate_model(y_val, xgb_scores, config)

    print("Training logistic regression (default settings)...")
    logreg_model, preprocessor = train_logistic_regression(train_df, config)
    X_val_logreg = preprocessor.transform(prepare_logreg_input(val_df))
    logreg_scores = logreg_model.predict_proba(X_val_logreg)[:, 1]
    logreg_metrics = evaluate_model(y_val, logreg_scores, config)

    results = {"xgboost_default": xgb_metrics, "logistic_regression": logreg_metrics}

    print()
    for name, metrics in results.items():
        print(f"--- {name} ---")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print()

    save_metrics(results, config)
    print(f"Saved metrics to {Path(config['paths']['metrics_dir']) / 'baseline.json'}")


if __name__ == "__main__":
    main()
