
"""Reproducibility helpers shared by preprocessing and model training."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedReport:
    seed: int
    python_seeded: bool
    numpy_seeded: bool
    tensorflow_seeded: bool


def set_global_seed(seed: int, *, include_tensorflow: bool = True) -> SeedReport:
    """Seed Python and any available numerical libraries."""

    if seed < 0:
        raise ValueError("seed must not be negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    numpy_seeded = False
    try:
        import numpy

        numpy.random.seed(seed)
        numpy_seeded = True
    except ImportError:
        pass

    tensorflow_seeded = False
    if include_tensorflow:
        try:
            import tensorflow

            tensorflow.keras.utils.set_random_seed(seed)
            try:
                tensorflow.config.experimental.enable_op_determinism()
            except (AttributeError, RuntimeError):
                pass
            tensorflow_seeded = True
        except ImportError:
            pass

    return SeedReport(
        seed=seed,
        python_seeded=True,
        numpy_seeded=numpy_seeded,
        tensorflow_seeded=tensorflow_seeded,
    )
