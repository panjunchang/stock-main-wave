"""分类评估指标工具。"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算二分类常用评估指标。

    参数
    ----
    y_true : array-like
        真实标签。
    y_pred : array-like
        预测标签。

    返回
    ----
    dict
        包含 accuracy、precision、recall、f1 的字典，值均为 float（保留 4 位小数）。
    """
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
