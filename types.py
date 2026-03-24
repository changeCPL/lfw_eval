"""评测用到的轻量数据结构（与具体模型、IO 解耦）。"""

from __future__ import annotations

from dataclasses import dataclass


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
