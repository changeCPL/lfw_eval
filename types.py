"""评测用到的轻量数据结构（与具体模型、IO 解耦）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryVerificationMetrics:
    """在某一阈值、某一子集（如单折测试）上的验证指标。"""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    tpr: float
    fpr: float
    tnr: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    n_genuine: int
    n_impostor: int


@dataclass(frozen=True)
class RocCurve:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray


@dataclass(frozen=True)
class PRCurve:
    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray


@dataclass(frozen=True)
class FacePair:
    """一对人脸样本在特征表中的键及是否同人。"""

    key_a: str
    key_b: str
    is_same: bool


@dataclass(frozen=True)
class TenFoldResult:
    """10 折交叉验证汇总结果。"""

    mean_accuracy: float
    std_accuracy: float
    fold_accuracies: tuple[float, ...]
    fold_thresholds: tuple[float, ...]
    #: 输入配对总数（与折划分长度一致，通常为 6000）
    total_pairs: int
    #: 两端特征均有效、且维度一致的配对数量
    valid_pair_count: int
    #: 每一折测试子集中有效配对数（分母）；用于核对缺失对分布
    fold_test_valid_counts: tuple[int, ...]
    #: 每一折在「该折训练得到的阈值」下，测试子集上的二分类指标
    fold_test_binary_metrics: tuple[BinaryVerificationMetrics, ...]
    #: 每折测试子集上的 ROC；仅 ``include_curves=True`` 时非空
    fold_roc: tuple[RocCurve, ...] | None
    #: 每折测试子集上的 PR 曲线；仅 ``include_curves=True`` 时非空
    fold_pr: tuple[PRCurve, ...] | None
