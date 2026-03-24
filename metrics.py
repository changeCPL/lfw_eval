"""
人脸验证（1:1）常用二分类指标与曲线。

约定：
- **正类（Positive）**：同人（genuine），label=1；判为同人当 ``score >= threshold``。
- **负类（Negative）**：异人（impostor），label=0。

由此：
- **TPR / Recall / TAR**：同人中被判为同人的比例。
- **FPR / FAR**：异人中被误判为同人的比例（与安全/冒名相关，用户体验上即「陌生人被当成你」）。
- **Precision**：判为同人的对里真正同人的比例（与「通过门禁的人里有多少是真的」相关）。
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .types import BinaryVerificationMetrics, PRCurve, RocCurve


def _mask_finite(valid: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return valid.astype(bool, copy=False) & np.isfinite(scores)


def confusion_counts_masked(
    scores: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    threshold: float,
) -> tuple[int, int, int, int]:
    """仅在 valid 位置上统计 TP/FP/TN/FN。"""
    v = valid.astype(bool, copy=False)
    y = labels.astype(np.int32)
    pred_pos = (scores >= threshold) & v
    y_pos = y == 1
    y_neg = y == 0
    tp = int(np.sum(pred_pos & y_pos))
    fp = int(np.sum(pred_pos & y_neg))
    fn = int(np.sum((~pred_pos) & v & y_pos))
    tn = int(np.sum((~pred_pos) & v & y_neg))
    return tp, fp, tn, fn


def binary_metrics_from_confusion(
    tp: int, fp: int, tn: int, fn: int
) -> BinaryVerificationMetrics:
    p = tp + fn
    n = fp + tn
    tpr = float(tp / p) if p > 0 else float("nan")
    fpr = float(fp / n) if n > 0 else float("nan")
    tnr = float(tn / n) if n > 0 else float("nan")
    denom_p = tp + fp
    precision = float(tp / denom_p) if denom_p > 0 else float("nan")
    recall = tpr
    if not np.isfinite(precision) or not np.isfinite(recall):
        f1 = float("nan")
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall / (precision + recall))
    tot = tp + fp + tn + fn
    acc = float((tp + tn) / tot) if tot > 0 else float("nan")
    return BinaryVerificationMetrics(
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        tpr=tpr,
        fpr=fpr,
        tnr=tnr,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=acc,
        n_genuine=p,
        n_impostor=n,
    )


def binary_metrics_at_threshold_masked(
    scores: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    threshold: float,
) -> BinaryVerificationMetrics:
    tp, fp, tn, fn = confusion_counts_masked(scores, labels, valid, threshold)
    return binary_metrics_from_confusion(tp, fp, tn, fn)


def roc_curve_masked(
    scores: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
) -> RocCurve:
    """
    ROC：横轴 FPR，纵轴 TPR；阈值从高到低扫描（分数越大越倾向判同人）。

    返回 ``(fpr, tpr, thresholds)``，首点 (0,0) 对应 ``threshold=+inf``，
    末点 (1,1) 对应 ``threshold=-inf``。
    """
    m = _mask_finite(valid, scores)
    s = scores[m].astype(np.float64)
    y = labels[m].astype(np.int32)
    p = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if p == 0 or n_neg == 0:
        return RocCurve(
            fpr=np.array([0.0, 1.0], dtype=np.float64),
            tpr=np.array([0.0, 1.0], dtype=np.float64),
            thresholds=np.array([np.inf, -np.inf], dtype=np.float64),
        )

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = np.concatenate([[0.0], tps.astype(np.float64) / p, [1.0]])
    fpr = np.concatenate([[0.0], fps.astype(np.float64) / n_neg, [1.0]])
    thr = np.concatenate([[np.inf], s_sorted, [-np.inf]])
    return RocCurve(fpr=fpr, tpr=tpr, thresholds=thr)


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    """梯形法积分；假定 ``x`` 非递减。"""
    if x.size < 2:
        return float("nan")
    _trapz = getattr(np, "trapezoid", np.trapz)
    return float(_trapz(y, x))


def precision_recall_curve_masked(
    scores: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
) -> PRCurve:
    """
    PR 曲线：横轴 Recall，纵轴 Precision；随阈值降低，Recall 上升、Precision 通常下降。

    ``thresholds[i]`` 对应第 i 个分界（与 sklearn 风格一致，长度比 precision 少 1 时可自行对齐绘图）。
    """
    m = _mask_finite(valid, scores)
    s = scores[m].astype(np.float64)
    y = labels[m].astype(np.int32)
    p = int(np.sum(y == 1))
    if p == 0:
        return PRCurve(
            precision=np.array([1.0], dtype=np.float64),
            recall=np.array([0.0], dtype=np.float64),
            thresholds=np.array([], dtype=np.float64),
        )

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tp_fp = tps + fps
    precision = tps.astype(np.float64) / np.maximum(tp_fp, 1)
    recall = tps.astype(np.float64) / p
    precision_ext = np.concatenate([[1.0], precision])
    recall_ext = np.concatenate([[0.0], recall])
    thr = s_sorted
    return PRCurve(
        precision=precision_ext,
        recall=recall_ext,
        thresholds=thr,
    )


def max_tpr_at_fpr_cap(
    roc: RocCurve,
    max_fpr: float,
    *,
    mode: Literal["leq", "closest"] = "leq",
) -> tuple[float, float]:
    """
    **TPR@FPR** 工作点（人脸中常写 TAR@FAR）。

    - ``leq``：在所有满足 ``FPR <= max_fpr`` 的 ROC 点上，取 **最大 TPR**；
      用于「把误识率压到不超过某值时，通过率能到多少」。
    - ``closest``：选 FPR 与 ``max_fpr`` 最接近且不超过或最接近的点（简化实现：仍优先 leq，若无则取最小 FPR 点）。

    返回 ``(tpr, threshold)``；若无有效点则 ``(nan, nan)``。
    """
    fpr = roc.fpr
    tpr = roc.tpr
    thr = roc.thresholds
    if fpr.size == 0:
        return float("nan"), float("nan")

    if mode == "leq":
        ok = fpr <= max_fpr + 1e-12
        if not np.any(ok):
            return float("nan"), float("nan")
        idx = int(np.argmax(np.where(ok, tpr, -1.0)))
        return float(tpr[idx]), float(thr[idx])

    # closest: 找 fpr 最接近 max_fpr 的点
    idx = int(np.argmin(np.abs(fpr - max_fpr)))
    return float(tpr[idx]), float(thr[idx])


def best_precision_at_min_recall(
    pr: PRCurve,
    min_recall: float,
) -> tuple[float, float]:
    """
    **Precision@Recall**：在满足 ``Recall >= min_recall`` 的 PR 点上取 **最大 Precision**。

    返回 ``(precision, recall)``；若无满足点则 ``(nan, nan)``。
    """
    prec = pr.precision
    rec = pr.recall
    if prec.size == 0:
        return float("nan"), float("nan")
    ok = rec >= min_recall - 1e-12
    if not np.any(ok):
        return float("nan"), float("nan")
    masked = np.where(ok, prec, -1.0)
    idx = int(np.argmax(masked))
    return float(prec[idx]), float(rec[idx])


def summarize_binary_metrics_across_folds(
    folds: tuple[BinaryVerificationMetrics, ...],
) -> dict[str, float]:
    """
    对多折在 **各自测试集、各自阈值** 下的标量指标做 ``nanmean``，便于汇报总体水平。

    注意：这与「合并全部测试分数画一条 ROC」不同；合并曲线更乐观，一般不作严格 CV。
    """
    keys = ("tpr", "fpr", "tnr", "precision", "recall", "f1", "accuracy")
    out: dict[str, float] = {}
    for name in keys:
        vals = np.array(
            [getattr(m, name) for m in folds],
            dtype=np.float64,
        )
        out[f"mean_{name}"] = float(np.nanmean(vals))
    return out


def average_precision_from_pr(pr: PRCurve) -> float:
    """
    由 PR 折线近似 Average Precision（与 sklearn 在稠密网格上略有数值差）。
    """
    p = pr.precision
    r = pr.recall
    if p.size < 2:
        return float("nan")
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))
