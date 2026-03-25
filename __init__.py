"""LFW 风格 10 折验证：解析配对行、中点阈值搜索、输出 mean/std。"""

from .evaluate import evaluate_10fold
from .metrics import (
    auc_trapezoid,
    average_precision_from_pr,
    best_precision_at_min_recall,
    binary_metrics_at_threshold_masked,
    max_tpr_at_fpr_cap,
    precision_recall_curve_masked,
    roc_curve_masked,
    summarize_binary_metrics_across_folds,
)
from .pair_parser import parse_pair_line, parse_pair_lines
from .overall import evaluate_whole_set_at_threshold, evaluate_whole_set_curves
from .similarity import pairs_to_scores_and_labels, pairs_to_scores_labels_valid
from .types import (
    BinaryVerificationMetrics,
    FacePair,
    PRCurve,
    RocCurve,
    TenFoldResult,
)

__all__ = [
    "BinaryVerificationMetrics",
    "FacePair",
    "PRCurve",
    "RocCurve",
    "TenFoldResult",
    "auc_trapezoid",
    "average_precision_from_pr",
    "best_precision_at_min_recall",
    "binary_metrics_at_threshold_masked",
    "evaluate_10fold",
    "evaluate_whole_set_at_threshold",
    "evaluate_whole_set_curves",
    "max_tpr_at_fpr_cap",
    "pairs_to_scores_and_labels",
    "pairs_to_scores_labels_valid",
    "parse_pair_line",
    "parse_pair_lines",
    "precision_recall_curve_masked",
    "roc_curve_masked",
    "summarize_binary_metrics_across_folds",
]
