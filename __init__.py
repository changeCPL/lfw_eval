"""LFW 风格 10 折验证：解析配对行、中点阈值搜索、输出 mean/std。"""

from .evaluate import evaluate_10fold
from .pair_parser import parse_pair_line, parse_pair_lines
from .similarity import pairs_to_scores_and_labels, pairs_to_scores_labels_valid
from .types import FacePair, TenFoldResult

__all__ = [
    "FacePair",
    "TenFoldResult",
    "evaluate_10fold",
    "pairs_to_scores_and_labels",
    "pairs_to_scores_labels_valid",
    "parse_pair_line",
    "parse_pair_lines",
]
