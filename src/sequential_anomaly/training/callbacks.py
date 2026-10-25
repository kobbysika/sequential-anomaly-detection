
"""Training callbacks shared by the modern sequence models."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_callbacks(
    output_directory: str | Path,
    *,
    patience: int = 64,
    monitor: str = "loss",
) -> list[Any]:
    """Create legacy-compatible early stopping plus durable training artifacts."""

    if patience < 0:
        raise ValueError("patience must not be negative")
    try:
        import tensorflow
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required to create training callbacks."
        ) from error

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    keras = tensorflow.keras
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output / "best.keras"),
            monitor=monitor,
            save_best_only=True,
        ),
        keras.callbacks.CSVLogger(str(output / "training.csv")),
    ]
