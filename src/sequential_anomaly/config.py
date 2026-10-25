
"""Portable experiment configuration for Colab and server execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathConfig:
    dataset_root: Path
    results_root: Path
    artifacts_root: Path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PathConfig:
        required = ("dataset_root", "results_root", "artifacts_root")
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise ValueError(f"Missing path settings: {', '.join(missing)}")
        return cls(**{key: Path(data[key]) for key in required})


@dataclass(frozen=True)
class PreprocessingConfig:
    window_minutes: int = 15
    sequence_length: int = 10
    keyset: int = 0
    packet_divisor: int = 100

    def __post_init__(self) -> None:
        if self.window_minutes <= 0 or 60 % self.window_minutes != 0:
            raise ValueError("window_minutes must be a positive divisor of 60")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero")
        if not 0 <= self.keyset <= 4:
            raise ValueError("keyset must be between 0 and 4")
        if self.packet_divisor <= 0:
            raise ValueError("packet_divisor must be greater than zero")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingConfig:
        return cls(**data)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    paths: PathConfig
    preprocessing: PreprocessingConfig

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Experiment name must not be blank")
        if self.seed < 0:
            raise ValueError("seed must not be negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        required = ("name", "seed", "paths", "preprocessing")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing configuration settings: {', '.join(missing)}")
        return cls(
            name=str(data["name"]),
            seed=int(data["seed"]),
            paths=PathConfig.from_dict(data["paths"]),
            preprocessing=PreprocessingConfig.from_dict(data["preprocessing"]),
        )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a JSON-compatible YAML configuration file."""

    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Cannot read configuration {source}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON-compatible YAML in {source}:{error.lineno}:{error.colno}: "
            f"{error.msg}"
        ) from error
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be an object: {source}")
    return ExperimentConfig.from_dict(data)
