"""
Phase 5: hyperparameter tuning with Optuna.

Tunes XGBoost's hyperparameters AND compares three ways of handling class
imbalance (none, scale_pos_weight, undersampling), by making the imbalance
method itself one of the things Optuna searches over -- one 50-trial budget
covering all three, rather than three separate 50-trial studies (which could
take up to 3x as long for the same answer, since Optuna's sampler naturally
spends more trials on whichever method tends to score better).

Every trial only uses TRAIN (to fit) and VALIDATION (to score). Test is
never loaded in this file -- see retrain_best_and_evaluate_once(), the only
function in the whole project that's allowed to touch it, and only ever
called once, at the very end.

Run from the project root, with the virtual environment active:
    python -m src.tune
"""

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna_integration import XGBoostPruningCallback

from src.config import load_config
from src.evaluate import evaluate_model, pr_auc, save_metrics
from src.train import TARGET_COL, load_feature_tables, prepare_xgb_data


def undersample_majority(train_df: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
    """
    Keep every laundering row, but randomly drop normal rows down to `ratio`
    normal-per-laundering (e.g. ratio=10 keeps 10 normal rows for every 1
    laundering row). This actually REMOVES data, unlike scale_pos_weight
    (which keeps every row but reweights how much each class counts in the
    loss function) -- that trade-off (less data vs. no data loss) is exactly
    what Phase 5 is comparing.
    """
    positives = train_df[train_df[TARGET_COL] == 1]
    negatives = train_df[train_df[TARGET_COL] == 0]
    n_keep = min(len(negatives), int(len(positives) * ratio))
    sampled_negatives = negatives.sample(n=n_keep, random_state=seed)
    combined = pd.concat([positives, sampled_negatives])
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)


def suggest_xgb_params(trial: optuna.Trial, search_space: dict) -> dict:
    """One XGBoost hyperparameter set drawn from the search space in config.yaml."""
    return {
        "max_depth": trial.suggest_int("max_depth", *search_space["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *search_space["learning_rate"], log=True),
        "n_estimators": trial.suggest_int("n_estimators", *search_space["n_estimators"]),
        "min_child_weight": trial.suggest_int("min_child_weight", *search_space["min_child_weight"]),
        "subsample": trial.suggest_float("subsample", *search_space["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *search_space["colsample_bytree"]),
        "gamma": trial.suggest_float("gamma", *search_space["gamma"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *search_space["reg_alpha"]),
        "reg_lambda": trial.suggest_float("reg_lambda", *search_space["reg_lambda"]),
    }


def make_objective(train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict):
    search_space = config["tuning"]["search_space"]
    seed = config["project"]["random_seed"]
    undersample_ratio = config["model"]["undersample_ratio"]

    X_val = prepare_xgb_data(val_df)
    y_val = val_df[TARGET_COL]

    def objective(trial: optuna.Trial) -> float:
        imbalance_method = trial.suggest_categorical("imbalance_method", ["none", "scale_pos_weight", "undersample"])
        params = suggest_xgb_params(trial, search_space)

        if imbalance_method == "undersample":
            working_train = undersample_majority(train_df, undersample_ratio, seed + trial.number)
        else:
            working_train = train_df

        if imbalance_method == "scale_pos_weight":
            params["scale_pos_weight"] = trial.suggest_float(
                "scale_pos_weight", *search_space["scale_pos_weight"], log=True
            )

        X_train = prepare_xgb_data(working_train)
        y_train = working_train[TARGET_COL]

        pruning_callback = XGBoostPruningCallback(trial, "validation_0-aucpr")
        model = xgb.XGBClassifier(
            **params,
            enable_categorical=True,
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=config["model"]["early_stopping_rounds"],
            random_state=seed,
            callbacks=[pruning_callback],
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        scores = model.predict_proba(X_val)[:, 1]
        return pr_auc(y_val, scores)

    return objective


def run_study(train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict) -> optuna.Study:
    tuning_cfg = config["tuning"]
    sampler = optuna.samplers.TPESampler(seed=config["project"]["random_seed"])
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        study_name=tuning_cfg["study_name"],
        storage=tuning_cfg["storage"],
        direction=tuning_cfg["direction"],
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,  # resumable: re-running this picks up where a previous run left off
    )
    objective = make_objective(train_df, val_df, config)
    study.optimize(
        objective,
        n_trials=tuning_cfg["n_trials"],
        timeout=tuning_cfg["timeout_minutes"] * 60,
    )
    return study


def summarize_imbalance_methods(study: optuna.Study) -> pd.DataFrame:
    """Best validation PR-AUC achieved by each imbalance method, across all completed trials."""
    rows = []
    for trial in study.trials:
        if trial.value is None:
            continue  # pruned or failed trial
        rows.append({"imbalance_method": trial.params["imbalance_method"], "pr_auc": trial.value})
    df = pd.DataFrame(rows)
    return df.groupby("imbalance_method")["pr_auc"].agg(["max", "mean", "count"]).sort_values("max", ascending=False)


def retrain_with_best_params(train_df: pd.DataFrame, val_df: pd.DataFrame, best_params: dict, config: dict) -> xgb.XGBClassifier:
    """Retrain a final model on train using the best trial's params, still only touching train+val."""
    params = {k: v for k, v in best_params.items() if k not in ("imbalance_method",)}
    imbalance_method = best_params["imbalance_method"]

    if imbalance_method == "undersample":
        working_train = undersample_majority(train_df, config["model"]["undersample_ratio"], config["project"]["random_seed"])
    else:
        working_train = train_df

    X_train = prepare_xgb_data(working_train)
    y_train = working_train[TARGET_COL]
    X_val = prepare_xgb_data(val_df)
    y_val = val_df[TARGET_COL]

    model = xgb.XGBClassifier(
        **params,
        enable_categorical=True,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=config["model"]["early_stopping_rounds"],
        random_state=config["project"]["random_seed"],
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def retrain_best_and_evaluate_once(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, best_params: dict, config: dict
) -> dict:
    """
    THE ONLY function in this whole project allowed to look at the test set,
    and only ever called once. Retrains with the tuned hyperparameters
    (chosen using train+val only, in run_study() above) and reports test
    metrics for the first and only time.
    """
    model = retrain_with_best_params(train_df, val_df, best_params, config)
    X_test = prepare_xgb_data(test_df)
    y_test = test_df[TARGET_COL]
    test_scores = model.predict_proba(X_test)[:, 1]
    return evaluate_model(y_test, test_scores, config)


def main() -> None:
    config = load_config()
    train_df, val_df = load_feature_tables(config)

    print(f"Starting Optuna study: {config['tuning']['n_trials']} trials, "
          f"{config['tuning']['timeout_minutes']} minute timeout, "
          f"resumable storage at {config['tuning']['storage']}")
    study = run_study(train_df, val_df, config)

    print(f"\nCompleted {len(study.trials)} trials.")
    print(f"Best validation PR-AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    print("\nImbalance method comparison (best PR-AUC per method across all trials):")
    print(summarize_imbalance_methods(study))

    # Test set loaded here for the first and only time in the whole project.
    features_dir = Path(config["paths"]["features_dir"])
    test_df = pd.read_parquet(features_dir / "test_features.parquet")

    print("\nRetraining with best params and evaluating on TEST (once, final check)...")
    test_metrics = retrain_best_and_evaluate_once(train_df, val_df, test_df, study.best_params, config)
    print(test_metrics)

    save_metrics({"xgboost_tuned_test": test_metrics, "best_params": study.best_params}, config, filename="tuned.json")
    print(f"\nSaved to {Path(config['paths']['metrics_dir']) / 'tuned.json'}")


if __name__ == "__main__":
    main()
