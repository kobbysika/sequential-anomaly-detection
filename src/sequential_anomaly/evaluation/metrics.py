
"""Dependency-free binary classification and ranking metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sequential_anomaly.evaluation.scoring import AnomalyScore


@dataclass(frozen=True)
class ConfusionMetrics:
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    specificity: float
    accuracy: float
    f1: float


@dataclass(frozen=True)
class CurvePoint:
    threshold: float
    true_positive_rate: float
    false_positive_rate: float
    precision: float
    recall: float
    f1: float


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def confusion_at_threshold(
    scores: list[AnomalyScore], threshold: float
) -> ConfusionMetrics:
    """Classify scores using the legacy strict `score > threshold` rule."""

    if not scores:
        raise ValueError("scores must not be empty")
    tp = fp = tn = fn = 0
    for item in scores:
        predicted = item.score > threshold
        if predicted and item.is_anomalous:
            tp += 1
        elif predicted:
            fp += 1
        elif item.is_anomalous:
            fn += 1
        else:
            tn += 1
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    return ConfusionMetrics(
        threshold=float(threshold),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        precision=precision,
        recall=recall,
        specificity=_divide(tn, tn + fp),
        accuracy=_divide(tp + tn, len(scores)),
        f1=_divide(2 * precision * recall, precision + recall),
    )


def ranking_curve(scores: list[AnomalyScore]) -> list[CurvePoint]:
    """Return ROC/PR points while lowering the anomaly threshold."""

    if not scores:
        raise ValueError("scores must not be empty")
    positives = sum(item.is_anomalous for item in scores)
    negatives = len(scores) - positives
    ordered_thresholds = sorted({item.score for item in scores}, reverse=True)
    thresholds = [1.0, *ordered_thresholds, -1.0]
    points = []
    for threshold in thresholds:
        result = confusion_at_threshold(scores, threshold)
        points.append(
            CurvePoint(
                threshold=threshold,
                true_positive_rate=_divide(result.true_positive, positives),
                false_positive_rate=_divide(result.false_positive, negatives),
                precision=(
                    result.precision
                    if result.true_positive + result.false_positive
                    else 1.0
                ),
                recall=result.recall,
                f1=result.f1,
            )
        )
    return points


def _trapezoid_area(points: list[tuple[float, float]]) -> float:
    return sum(
        (right_x - left_x) * (left_y + right_y) / 2
        for (left_x, left_y), (right_x, right_y) in zip(points, points[1:])
    )


def evaluate_scores(
    scores: list[AnomalyScore], *, threshold: float = 0.05
) -> dict[str, object]:
    """Compute report-ready point metrics and ROC/PR summaries."""

    curve = ranking_curve(scores)
    roc_points = [(point.false_positive_rate, point.true_positive_rate) for point in curve]
    pr_points = sorted(
        ((point.recall, point.precision) for point in curve), key=lambda point: point[0]
    )
    auroc = _trapezoid_area(roc_points)
    auprc = _trapezoid_area(pr_points)
    best = max(curve, key=lambda point: point.f1)
    return {
        "groups": len(scores),
        "anomalous_groups": sum(item.is_anomalous for item in scores),
        "normal_groups": sum(not item.is_anomalous for item in scores),
        "at_threshold": asdict(confusion_at_threshold(scores, threshold)),
        "auroc": auroc,
        "auprc": auprc,
        "best_f1": best.f1,
        "best_f1_threshold": best.threshold,
        "curve": [asdict(point) for point in curve],
    }
