"""10 折交叉验证：每折在 9 份上选中点法最优阈值，在剩余 1 份上报告准确率。"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .similarity import pairs_to_scores_and_labels
from .threshold import accuracy_at_threshold, select_threshold_max_train_accuracy
from .types import FacePair, TenFoldResult


def evaluate_10fold(
    features: Mapping[str, np.ndarray],
    pairs: list[FacePair],
    *,
    n_folds: int = 10,
) -> TenFoldResult:
    """
    假定 ``pairs`` 的顺序与 LFW View-2 一致：总长 ``n_folds * fold_size``，
    第 ``k`` 折测试区间为 ``[k * fold_size, (k+1) * fold_size)``，其余为训练。

    「10 段连续块」指：第 0 折对应列表中下标 ``0 .. fold_size-1``，第 1 折对应
    ``fold_size .. 2*fold_size-1``，依此类推；每一段内部是 300 正 + 300 负或打乱
    均可，只要整段属于同一折即可。

    若你的文件是「表头 + (300 同人 + 300 异人) 重复 10 次」，解析后共 6000 对且
    顺序不变，则与上述分折一致。若整文件是「表头 + 3000 同人 + 3000 异人」而
    未按折重排，则前 600 对全是同人，与标准 10 折块划分不一致，需先按官方折
    顺序重排或改用显式折索引（可另行扩展 API）。

    若长度不能整除 ``n_folds``，抛出 ValueError。
    """
    n = len(pairs)
    if n_folds <= 0:
        raise ValueError("n_folds 必须为正整数")
    if n % n_folds != 0:
        raise ValueError(
            f"pairs 长度 {n} 不能被折数 {n_folds} 整除；"
            "请确认顺序是否为连续的 n_folds 个等长折块。"
        )
    fold_size = n // n_folds

    scores, labels = pairs_to_scores_and_labels(features, pairs)

    fold_accs: list[float] = []
    fold_ts: list[float] = []

    for k in range(n_folds):
        lo, hi = k * fold_size, (k + 1) * fold_size
        test_mask = np.zeros(n, dtype=bool)
        test_mask[lo:hi] = True
        train_mask = ~test_mask

        scores_tr = scores[train_mask]
        labels_tr = labels[train_mask]
        scores_te = scores[test_mask]
        labels_te = labels[test_mask]

        t = select_threshold_max_train_accuracy(scores_tr, labels_tr)
        acc_te = accuracy_at_threshold(scores_te, labels_te, t)
        fold_accs.append(acc_te)
        fold_ts.append(t)

    arr = np.asarray(fold_accs, dtype=np.float64)
    mean_acc = float(arr.mean())
    std_acc = float(arr.std(ddof=1)) if arr.size > 1 else 0.0

    return TenFoldResult(
        mean_accuracy=mean_acc,
        std_accuracy=std_acc,
        fold_accuracies=tuple(fold_accs),
        fold_thresholds=tuple(fold_ts),
    )
