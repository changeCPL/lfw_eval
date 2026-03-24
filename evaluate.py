"""10 折交叉验证：每折在 9 份上选中点法最优阈值，在剩余 1 份上报告准确率。"""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np

from .metrics import (
    binary_metrics_at_threshold_masked,
    precision_recall_curve_masked,
    roc_curve_masked,
)
from .similarity import pairs_to_scores_and_labels, pairs_to_scores_labels_valid
from .threshold import (
    accuracy_at_threshold,
    accuracy_at_threshold_masked,
    select_threshold_max_train_accuracy,
    select_threshold_max_train_accuracy_masked,
)
from .types import FacePair, TenFoldResult


def evaluate_10fold(
    features: Mapping[str, np.ndarray],
    pairs: list[FacePair],
    *,
    n_folds: int = 10,
    on_missing: Literal["raise", "mask"] = "raise",
    include_curves: bool = False,
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

    ``on_missing``:
        - ``raise``：任一对缺特征或维度不一致则报错（与旧行为一致）。
        - ``mask``：保留长度 ``n`` 与折下标；无效对不参与阈值搜索与准确率分母，
          折划分仍按原始下标，避免「删掉部分对后长度非 6000」导致无法分折。

    ``include_curves``：为 True 时，在**每一折的测试子集**上额外计算 ROC 与 PR 曲线
    （用于 TPR@FPR、AUC、Precision@Recall 等工作点或绘图）；折与折之间曲线独立，
    若需「全数据一条 ROC」请自行合并分数（注意与 CV 口径不同）。

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

    if on_missing == "raise":
        scores, labels = pairs_to_scores_and_labels(features, pairs)
        valid = np.ones(n, dtype=bool)
    else:
        scores, labels, valid = pairs_to_scores_labels_valid(features, pairs)

    fold_accs: list[float] = []
    fold_ts: list[float] = []
    fold_test_valid: list[int] = []
    fold_bin: list = []
    fold_rocs: list = []
    fold_prs: list = []

    for k in range(n_folds):
        lo, hi = k * fold_size, (k + 1) * fold_size
        test_mask = np.zeros(n, dtype=bool)
        test_mask[lo:hi] = True
        train_mask = ~test_mask

        valid_tr = train_mask & valid
        valid_te = test_mask & valid
        n_te = int(np.count_nonzero(valid_te))
        fold_test_valid.append(n_te)

        if on_missing == "raise":
            scores_tr = scores[train_mask]
            labels_tr = labels[train_mask]
            scores_te = scores[test_mask]
            labels_te = labels[test_mask]
            t = select_threshold_max_train_accuracy(scores_tr, labels_tr)
            acc_te = accuracy_at_threshold(scores_te, labels_te, t)
        else:
            if not np.any(valid_tr):
                raise ValueError(
                    f"第 {k} 折：训练侧（其余 9 折）无有效配对，无法搜索阈值"
                )
            t = select_threshold_max_train_accuracy_masked(
                scores, labels, valid_tr
            )
            acc_te = accuracy_at_threshold_masked(scores, labels, valid_te, t)

        fold_accs.append(acc_te)
        fold_ts.append(t)

        bm = binary_metrics_at_threshold_masked(scores, labels, valid_te, t)
        fold_bin.append(bm)

        if include_curves:
            fold_rocs.append(roc_curve_masked(scores, labels, valid_te))
            fold_prs.append(precision_recall_curve_masked(scores, labels, valid_te))

    arr = np.asarray(fold_accs, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        mean_acc = float("nan")
        std_acc = float("nan")
    else:
        mean_acc = float(np.nanmean(arr))
        std_acc = (
            float(np.nanstd(arr, ddof=1))
            if int(np.sum(finite)) > 1
            else 0.0
        )

    valid_count = int(np.count_nonzero(valid))

    return TenFoldResult(
        mean_accuracy=mean_acc,
        std_accuracy=std_acc,
        fold_accuracies=tuple(fold_accs),
        fold_thresholds=tuple(fold_ts),
        total_pairs=n,
        valid_pair_count=valid_count,
        fold_test_valid_counts=tuple(fold_test_valid),
        fold_test_binary_metrics=tuple(fold_bin),
        fold_roc=tuple(fold_rocs) if include_curves else None,
        fold_pr=tuple(fold_prs) if include_curves else None,
    )
