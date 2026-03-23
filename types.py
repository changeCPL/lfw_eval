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
