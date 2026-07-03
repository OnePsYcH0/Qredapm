from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


def _validate_binary_inputs(y_true: Iterable[float], y_score: Iterable[float]) -> Tuple[List[int], List[float]]:
    true_values = [int(v) for v in y_true]
    score_values = [float(v) for v in y_score]

    if not true_values:
        raise ValueError("空标签序列，无法计算指标。")
    if len(true_values) != len(score_values):
        raise ValueError("标签数量与预测分数数量不一致。")
    if any(v not in (0, 1) for v in true_values):
        raise ValueError("只支持二分类标签 0/1。")
    if len(set(true_values)) < 2:
        raise ValueError("ROC-AUC 和 PR-AUC 至少需要同时存在正负样本。")
    return true_values, score_values


def _confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[int, int, int, int]:
    tn = fp = fn = tp = 0
    for truth, pred in zip(y_true, y_pred):
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 1 and pred == 0:
            fn += 1
        elif truth == 0 and pred == 1:
            fp += 1
        else:
            tn += 1
    return tn, fp, fn, tp


def _roc_auc_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    paired = sorted(zip(y_score, y_true), key=lambda item: item[0])
    positive_total = sum(y_true)
    negative_total = len(y_true) - positive_total

    rank_sum_positive = 0.0
    rank = 1
    index = 0
    total = len(paired)
    while index < total:
        next_index = index
        while next_index < total and paired[next_index][0] == paired[index][0]:
            next_index += 1

        group_size = next_index - index
        avg_rank = (rank + rank + group_size - 1) / 2.0
        positive_count = sum(label for _, label in paired[index:next_index])
        rank_sum_positive += avg_rank * positive_count

        rank += group_size
        index = next_index

    return (rank_sum_positive - positive_total * (positive_total + 1) / 2.0) / (positive_total * negative_total)


def _pr_curve_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    paired = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
    positive_total = sum(y_true)
    precisions: List[float] = []
    recalls: List[float] = []
    tp = 0
    fp = 0
    index = 0
    total = len(paired)

    while index < total:
        current_score = paired[index][0]
        while index < total and paired[index][0] == current_score:
            if paired[index][1] == 1:
                tp += 1
            else:
                fp += 1
            index += 1

        precisions.append(tp / (tp + fp))
        recalls.append(tp / positive_total)

    precisions = [1.0] + precisions
    recalls = [0.0] + recalls

    area = 0.0
    for left_recall, right_recall, left_precision, right_precision in zip(
        recalls[:-1],
        recalls[1:],
        precisions[:-1],
        precisions[1:],
    ):
        area += (right_recall - left_recall) * (left_precision + right_precision) / 2.0
    return area


def compute_binary_metrics(
    y_true: Iterable[float],
    y_score: Iterable[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true_list, y_score_list = _validate_binary_inputs(y_true, y_score)
    y_pred_list = [1 if score >= threshold else 0 for score in y_score_list]
    tn, fp, fn, tp = _confusion_counts(y_true_list, y_pred_list)
    total = len(y_true_list)
    positive_total = sum(y_true_list)
    negative_total = total - positive_total

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "threshold": float(threshold),
        "sample_count": int(total),
        "positive_count": int(positive_total),
        "negative_count": int(negative_total),
        "roc_auc": float(_roc_auc_score(y_true_list, y_score_list)),
        "pr_auc": float(_pr_curve_auc(y_true_list, y_score_list)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
