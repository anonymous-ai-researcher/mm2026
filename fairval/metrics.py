"""Evaluation metrics for fair visual active learning."""

import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from typing import Optional


def compute_tpr_per_group(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Compute per-group true positive rates.

    Returns:
        (k,) tensor of TPR values.
    """
    tprs = torch.zeros(num_groups)
    for a in range(num_groups):
        mask = (groups == a) & (labels == 1)
        if mask.sum() > 0:
            tprs[a] = predictions[mask].float().mean()
        else:
            tprs[a] = float("nan")
    return tprs


def compute_eod(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    num_groups: int,
) -> float:
    """Equal Opportunity Difference: max_{a,a'} |TPR_a - TPR_{a'}|.

    Args:
        predictions: (N,) binary predictions.
        labels: (N,) true binary labels.
        groups: (N,) group assignments.
        num_groups: Number of groups (k).

    Returns:
        Scalar EOD value.
    """
    tprs = compute_tpr_per_group(predictions, labels, groups, num_groups)
    valid = ~tprs.isnan()
    if valid.sum() < 2:
        return 0.0
    valid_tprs = tprs[valid]
    return (valid_tprs.max() - valid_tprs.min()).item()


def compute_wgr(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    num_groups: int,
) -> float:
    """Worst-group recall (minimum TPR across groups).

    Returns:
        Scalar WGR value.
    """
    tprs = compute_tpr_per_group(predictions, labels, groups, num_groups)
    valid = ~tprs.isnan()
    if valid.sum() == 0:
        return 0.0
    return tprs[valid].min().item()


def compute_f1(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Macro F1 score.

    Returns:
        Scalar F1 value.
    """
    preds_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()
    return f1_score(labels_np, preds_np, average="macro", zero_division=0)


def compute_auroc(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Area under the ROC curve.

    Returns:
        Scalar AUROC value.
    """
    probs_np = probabilities.cpu().numpy()
    labels_np = labels.cpu().numpy()
    try:
        return roc_auc_score(labels_np, probs_np)
    except ValueError:
        return 0.5


def labels_to_target(
    eod_history: list[float],
    gamma: float,
    labels_per_round: int,
) -> Optional[int]:
    """Compute labels-to-target: first round where EOD ≤ γ.

    Args:
        eod_history: List of EOD values per round.
        gamma: Fairness tolerance.
        labels_per_round: Number of labels acquired per round (m).

    Returns:
        Total labels when target is first met, or None if never met.
    """
    for t, eod in enumerate(eod_history):
        if eod <= gamma:
            return (t + 1) * labels_per_round
    return None


def full_evaluation(
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    num_groups: int,
) -> dict:
    """Run all metrics and return a summary dictionary."""
    return {
        "f1": compute_f1(predictions, labels),
        "auroc": compute_auroc(probabilities, labels),
        "eod": compute_eod(predictions, labels, groups, num_groups),
        "wgr": compute_wgr(predictions, labels, groups, num_groups),
        "tpr_per_group": compute_tpr_per_group(
            predictions, labels, groups, num_groups
        ).tolist(),
    }
