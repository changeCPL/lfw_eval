"""由特征表与配对列表计算相似度与标签（与分折、阈值搜索解耦）。"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .types import FacePair


def _get_vec_optional(
    features: Mapping[str, np.ndarray], key: str
) -> np.ndarray | None:
    if key not in features:
        return None
    return np.asarray(features[key], dtype=np.float64).ravel()


def _get_vec(features: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    v = _get_vec_optional(features, key)
    if v is None:
        raise KeyError(f"特征表中缺少键: {key!r}")
    return v


def pairs_to_scores_labels_valid(
    features: Mapping[str, np.ndarray], pairs: list[FacePair]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对每一对计算点积相似度（假定特征已 L2 归一化），并给出是否可评测。

    任一端缺键或维度不一致时：``valid[i]=False``，``scores[i]=nan``，``labels[i]``
    仍保留真值标签（便于统计缺失分布）；评测时必须用 ``valid`` 掩码排除。

    返回:
        scores: shape (N,)，无效位置为 nan
        labels: shape (N,)，同人=1，异人=0
        valid: shape (N,) bool，两端特征存在且维度一致
    """
    if not pairs:
        z = np.zeros(0, dtype=np.float64)
        return z, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=bool)

    scores_list: list[float] = []
    labels_list: list[int] = []
    valid_list: list[bool] = []
    for p in pairs:
        label = 1 if p.is_same else 0
        va = _get_vec_optional(features, p.key_a)
        vb = _get_vec_optional(features, p.key_b)
        if va is None or vb is None or va.shape != vb.shape:
            scores_list.append(float("nan"))
            labels_list.append(label)
            valid_list.append(False)
            continue
        scores_list.append(float(np.dot(va, vb)))
        labels_list.append(label)
        valid_list.append(True)

    scores = np.asarray(scores_list, dtype=np.float64)
    labels = np.asarray(labels_list, dtype=np.int32)
    valid = np.asarray(valid_list, dtype=bool)
    return scores, labels, valid


def pairs_to_scores_and_labels(
    features: Mapping[str, np.ndarray], pairs: list[FacePair]
) -> tuple[np.ndarray, np.ndarray]:
    """
    对每一对计算点积相似度（假定特征已 L2 归一化）。

    若存在缺键或维度不一致，抛出 ``KeyError`` / ``ValueError``。
    若需保留 6000 槽位并跳过无效对，请用 ``pairs_to_scores_labels_valid``。
    """
    scores, labels, valid = pairs_to_scores_labels_valid(features, pairs)
    if not bool(np.all(valid)):
        bad = int(np.size(valid) - int(np.count_nonzero(valid)))
        raise ValueError(
            f"存在 {bad} 对无法计算相似度（缺特征或维度不一致）；"
            "请使用 evaluate_10fold(..., on_missing='mask') 或 pairs_to_scores_labels_valid。"
        )
    return scores, labels
