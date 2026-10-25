
"""Report-ready plots for sequence anomaly evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_evaluation_plots(metrics: dict[str, Any], output_directory: str | Path) -> None:
    """Save ROC and precision-recall curves as publication-quality PNG files."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError as error:
        raise RuntimeError("Matplotlib is required to create evaluation plots") from error

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    curve = metrics["curve"]

    figure, axis = pyplot.subplots(figsize=(5, 4), constrained_layout=True)
    axis.plot(
        [point["false_positive_rate"] for point in curve],
        [point["true_positive_rate"] for point in curve],
        label=f"AUROC = {metrics['auroc']:.3f}",
    )
    axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    axis.legend(loc="lower right")
    figure.savefig(output / "roc_curve.png", dpi=200)
    pyplot.close(figure)

    pr_curve = sorted(curve, key=lambda point: point["recall"])
    figure, axis = pyplot.subplots(figsize=(5, 4), constrained_layout=True)
    axis.plot(
        [point["recall"] for point in pr_curve],
        [point["precision"] for point in pr_curve],
        label=f"AUPRC = {metrics['auprc']:.3f}",
    )
    axis.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall Curve")
    axis.legend(loc="lower left")
    figure.savefig(output / "precision_recall_curve.png", dpi=200)
    pyplot.close(figure)
