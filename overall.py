"""整数据集评测接口（不依赖 fold 划分）。"""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np

from .metrics import (
    auc_trapezoid,
    average_precision_from_pr,
    best_precision_at_min_recall,
    binary_metrics_at_threshold_masked,
    max_tpr_at_fpr_cap,
    precision_recall_curve_masked,
    roc_curve_masked,
)
from .similarity import pairs_to_scores_and_labels, pairs_to_scores_labels_valid
from .types import BinaryVerificationMetrics, FacePair, PRCurve, RocCurve


def _scores_labels_valid_from_pairs(
    features: Mapping[str, np.ndarray],
    pairs: list[FacePair],
    *,
    on_missing: Literal["raise", "mask"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """统一获取 scores/labels/valid，便于整数据集评测复用。"""
    n = len(pairs)
    if on_missing == "raise":
        scores, labels = pairs_to_scores_and_labels(features, pairs)
        valid = np.ones(n, dtype=bool)
        return scores, labels, valid
    return pairs_to_scores_labels_valid(features, pairs)


def evaluate_whole_set_curves(
    features: Mapping[str, np.ndarray],
    pairs: list[FacePair],
    *,
    on_missing: Literal["raise", "mask"] = "mask",
    fpr_caps: tuple[float, ...] = (),
    min_recalls: tuple[float, ...] = (),
) -> dict[str, object]:
    """
    在整个数据集上计算曲线类指标与工作点（不做 fold 划分）。

    返回字段：
        - total_pairs, valid_pair_count
        - roc, pr
        - roc_auc, ap
        - tpr_at_fpr: dict[cap -> (tpr, fpr, threshold)]
        - precision_at_recall: dict[min_recall -> (precision, recall, threshold)]
    """
    scores, labels, valid = _scores_labels_valid_from_pairs(
        features, pairs, on_missing=on_missing
    )
    roc: RocCurve = roc_curve_masked(scores, labels, valid)
    pr: PRCurve = precision_recall_curve_masked(scores, labels, valid)
    roc_auc = auc_trapezoid(roc.fpr, roc.tpr)
    ap = average_precision_from_pr(pr)

    tpr_at_fpr: dict[float, tuple[float, float, float]] = {}
    for cap in fpr_caps:
        tpr_at_fpr[float(cap)] = max_tpr_at_fpr_cap(roc, float(cap))

    precision_at_recall: dict[float, tuple[float, float, float]] = {}
    for r in min_recalls:
        precision_at_recall[float(r)] = best_precision_at_min_recall(pr, float(r))

    return {
        "total_pairs": len(pairs),
        "valid_pair_count": int(np.count_nonzero(valid)),
        "roc": roc,
        "pr": pr,
        "roc_auc": float(roc_auc),
        "ap": float(ap),
        "tpr_at_fpr": tpr_at_fpr,
        "precision_at_recall": precision_at_recall,
    }


def evaluate_whole_set_at_threshold(
    features: Mapping[str, np.ndarray],
    pairs: list[FacePair],
    threshold: float,
    *,
    on_missing: Literal["raise", "mask"] = "mask",
) -> dict[str, object]:
    """
    在整个数据集上，按给定阈值计算二分类指标。

    返回字段：
        - total_pairs, valid_pair_count
        - threshold
        - metrics: BinaryVerificationMetrics
    """
    scores, labels, valid = _scores_labels_valid_from_pairs(
        features, pairs, on_missing=on_missing
    )
    metrics: BinaryVerificationMetrics = binary_metrics_at_threshold_masked(
        scores=scores, labels=labels, valid=valid, threshold=float(threshold)
    )
    return {
        "total_pairs": len(pairs),
        "valid_pair_count": int(np.count_nonzero(valid)),
        "threshold": float(threshold),
        "metrics": metrics,
    }

