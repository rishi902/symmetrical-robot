"""
Phase 4: evaluation metrics for the laundering-detection models.

Accuracy is NOT used here. With laundering at ~0.1% of transactions, a model
that predicts "normal" for everything scores ~99.9% accuracy while catching
zero laundering -- accuracy can't tell a useless model from a good one when
the classes are this imbalanced. Every metric below is chosen specifically
to look at how well the model handles the rare (laundering) class.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def pr_auc(y_true, y_scores) -> float:
    """
    PR-AUC (average precision): area under the precision-recall curve.
    The PRIMARY metric for this project. Unlike ROC-AUC, it isn't inflated
    by a huge number of easy true negatives, which matters a lot here: for
    every possible number of alerts raised, it looks at how many are
    correct and how much of the real laundering gets caught -- focused
    entirely on the rare positive class.
    """
    return float(average_precision_score(y_true, y_scores))


def roc_auc(y_true, y_scores) -> float:
    """
    ROC-AUC: the probability the model ranks a random laundering
    transaction above a random normal one. Reported for reference, but NOT
    the main metric -- with 99.9% of transactions being "normal", it's easy
    to correctly rank the huge number of obvious true negatives, so this
    number can look deceptively good even for a mediocre model.
    """
    return float(roc_auc_score(y_true, y_scores))


def recall_at_fixed_fpr(y_true, y_scores, target_fpr: float) -> float:
    """
    A concrete, operational question: "if analysts can only tolerate a
    false-positive rate of `target_fpr` (e.g. 1% of all normal transactions
    wrongly flagged), how much of the real laundering do we actually
    catch?" More useful to a bank than a single abstract AUC number.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    achievable = fpr <= target_fpr
    if not achievable.any():
        return 0.0
    return float(tpr[achievable].max())


def classification_metrics(y_true, y_pred_hard) -> dict:
    """Precision, recall, F1 for the laundering (positive) class at a chosen threshold."""
    return {
        "precision": float(precision_score(y_true, y_pred_hard, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_hard, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_hard, zero_division=0)),
    }


def business_cost(y_true, y_pred_hard, cost_false_negative: float, cost_false_positive: float) -> float:
    """
    total cost = (missed laundering cases x cost of missing one)
               + (false alerts x cost of an analyst reviewing one)

    ASSUMPTION, not a real bank figure -- see config.yaml and the README's
    honesty note. Used only to compare models/thresholds against each
    other, not as a real financial estimate.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_hard, labels=[0, 1]).ravel()
    return float(fn * cost_false_negative + fp * cost_false_positive)


def choose_threshold_by_business_cost(
    y_true, y_scores, cost_false_negative: float, cost_false_positive: float
) -> tuple[float, float]:
    """
    Find the decision threshold that minimises total business cost, chosen
    on whatever data is passed in -- always call this with VALIDATION data,
    never test (test is touched exactly once, at the very end of Phase 5).

    Implemented as a single sort + cumulative sum rather than testing every
    threshold independently (which would be O(n^2) and too slow on ~90k+
    rows): sort transactions by score descending, then sweep the cutoff
    down the sorted list one transaction at a time, tracking how many true
    positives and false positives have been flagged so far at each point.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_positive = int(y_true.sum())

    order = np.argsort(-y_scores)
    sorted_scores = y_scores[order]
    sorted_labels = y_true[order]

    cum_tp = np.cumsum(sorted_labels)
    cum_fp = np.cumsum(1 - sorted_labels)

    fn = n_positive - cum_tp
    fp = cum_fp
    cost = fn * cost_false_negative + fp * cost_false_positive

    best_idx = int(np.argmin(cost))
    return float(sorted_scores[best_idx]), float(cost[best_idx])


def evaluate_model(y_true, y_scores, config: dict) -> dict:
    """Compute every metric for one model's predictions, and pick a business-cost-minimising threshold."""
    eval_cfg = config["evaluation"]
    costs = eval_cfg["costs"]

    threshold, cost_at_threshold = choose_threshold_by_business_cost(
        y_true, y_scores, costs["false_negative_usd"], costs["false_positive_usd"]
    )
    y_pred_hard = (np.asarray(y_scores) >= threshold).astype(int)

    return {
        "pr_auc": pr_auc(y_true, y_scores),
        "roc_auc": roc_auc(y_true, y_scores),
        "recall_at_fixed_fpr": recall_at_fixed_fpr(y_true, y_scores, eval_cfg["fixed_fpr"]),
        "chosen_threshold": threshold,
        "business_cost_usd": cost_at_threshold,
        **classification_metrics(y_true, y_pred_hard),
    }


def save_metrics(metrics: dict, config: dict, filename: str = "baseline.json") -> None:
    import json

    metrics_dir = Path(config["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / filename).write_text(json.dumps(metrics, indent=2))
