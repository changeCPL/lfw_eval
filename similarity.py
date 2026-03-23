"""由特征表与配对列表计算相似度与标签（与分折、阈值搜索解耦）。"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .types import FacePair


def _get_vec(features: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in features:
        raise KeyError(f"特征表中缺少键: {key!r}")
    v = np.asarray(features[key], dtype=np.float64).ravel()
    return v


def pairs_to_scores_and_labels(
    features: Mapping[str, np.ndarray], pairs: list[FacePair]
) -> tuple[np.ndarray, np.ndarray]:
    """
    对每一对计算点积相似度（假定特征已 L2 归一化）。

    返回:
        scores: shape (N,)
        labels: shape (N,), 同人=1，异人=0
    """
    if not pairs:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int32)

    scores_list: list[float] = []
    labels_list: list[int] = []
    for p in pairs:
        va = _get_vec(features, p.key_a)
        vb = _get_vec(features, p.key_b)
        if va.shape != vb.shape:
            raise ValueError(
                f"键 {p.key_a!r} 与 {p.key_b!r} 的特征维度不一致: "
                f"{va.shape} vs {vb.shape}"
            )
        scores_list.append(float(np.dot(va, vb)))
        labels_list.append(1 if p.is_same else 0)

    scores = np.asarray(scores_list, dtype=np.float64)
    labels = np.asarray(labels_list, dtype=np.int32)
    return scores, labels
