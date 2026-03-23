"""基于训练折相似度分数的阈值搜索（中点法，与常见 LFW 实现一致）。"""

from __future__ import annotations

import numpy as np


def midpoint_threshold_candidates(scores: np.ndarray) -> np.ndarray:
    """
    生成候选阈值集合（中点法）。

    将训练折上所有分数去重排序后，在相邻两个不同分数之间取中点作为候选；
    并在最小分数左侧、最大分数右侧各加一个边界候选，避免极端全判同人/全判异人
    时无法被中点覆盖的情况。

    对 L2 归一化特征，相似度一般为点积，越大越像同人：判定规则为 ``score >= t``。
    """
    if scores.size == 0:
        return np.array([0.0], dtype=np.float64)

    u = np.sort(np.unique(scores.astype(np.float64)))
    if u.size == 1:
        eps = 1e-6
        return np.array([float(u[0] - eps), float(u[0]), float(u[0] + eps)])

    mids = (u[:-1] + u[1:]) / 2.0
    eps = 1e-6
    lo = float(u[0] - eps)
    hi = float(u[-1] + eps)
    return np.concatenate(([lo], mids.astype(np.float64), [hi]))


def accuracy_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> float:
    """
    labels: 同人=1，异人=0。
    预测同人当且仅当 ``score >= threshold``。
    """
    pred = (scores >= threshold).astype(np.int32)
    return float((pred == labels).mean())


def select_threshold_max_train_accuracy(
    scores_train: np.ndarray, labels_train: np.ndarray
) -> float:
    """
    在训练折上枚举「中点法」候选阈值，选取使训练准确率最高的阈值。

    若有多个阈值并列最优，取其中数值最小的一个，保证结果可复现。
    """
    candidates = midpoint_threshold_candidates(scores_train)
    best_t = float(candidates[0])
    best_acc = -1.0
    for t in candidates:
        acc = accuracy_at_threshold(scores_train, labels_train, float(t))
        if acc > best_acc + 1e-15:
            best_acc = acc
            best_t = float(t)
    return best_t
