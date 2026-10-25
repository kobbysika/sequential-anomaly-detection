
"""Shared, reproducible training pipeline for DabLog and DeepLog."""

from __future__ import annotations

import json
import platform
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sequential_anomaly.models.dablog import DablogModelConfig, build_dablog
from sequential_anomaly.models.deeplog import DeeplogModelConfig, build_deeplog
from sequential_anomaly.reproducibility import set_global_seed
from sequential_anomaly.training.callbacks import build_callbacks


@dataclass(frozen=True)
class TrainingConfig:
    model: str
    vocabulary_size: int
    sequence_length: int = 10
    epochs: int = 32
    batch_size: int | None = None
    patience: int = 64
    learning_rate: float = 0.001

    def __post_init__(self) -> None:
        if self.model not in {"dablog", "deeplog"}:
            raise ValueError("model must be 'dablog' or 'deeplog'")
        if self.vocabulary_size <= 4:
            raise ValueError("vocabulary_size must be greater than four")
        if self.sequence_length <= 0 or self.epochs <= 0:
            raise ValueError("sequence_length and epochs must be greater than zero")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.patience < 0 or self.learning_rate <= 0:
            raise ValueError("patience must be nonnegative and learning_rate positive")

    @property
    def effective_batch_size(self) -> int:
        if self.batch_size is not None:
            return self.batch_size
        return 256 if self.model == "dablog" else 2048


def read_prepared_windows(path: str | Path) -> list[dict[str, Any]]:
    """Read and minimally validate JSONL emitted by prepare_data."""

    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
                event_ids = record["event_ids"]
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"Invalid prepared record at {path}:{line_number}") from error
            if not isinstance(event_ids, list) or not all(
                isinstance(value, int) and value >= 0 for value in event_ids
            ):
                raise ValueError(f"Invalid event_ids at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"Prepared training data is empty: {path}")
    return records


def dablog_examples(
    records: Iterable[dict[str, Any]], *, sequence_length: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Return inputs and reverse-sequence targets used by DabLog."""

    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for record in records:
        events = list(record["event_ids"])
        if len(events) != sequence_length:
            raise ValueError("DabLog records must match the configured sequence length")
        inputs.append(events)
        target = record.get("reversed_target_ids", list(reversed(events)))
        if len(target) != sequence_length:
            raise ValueError("DabLog targets must match the configured sequence length")
        targets.append(list(target))
    return inputs, targets


def reconstruct_group_sequences(
    records: Iterable[dict[str, Any]], *, pad_id: int = 0
) -> list[list[int]]:
    """Undo overlapping DabLog windows to recover each legacy event group."""

    groups: OrderedDict[str, list[list[int]]] = OrderedDict()
    for record in records:
        group_id = record.get("group_id")
        if not isinstance(group_id, str):
            raise ValueError("Every prepared record must contain a string group_id")
        groups.setdefault(group_id, []).append(list(record["event_ids"]))

    sequences: list[list[int]] = []
    for windows in groups.values():
        first = [event_id for event_id in windows[0] if event_id != pad_id]
        sequence = first + [window[-1] for window in windows[1:]]
        sequences.append(sequence)
    return sequences


def deeplog_examples(
    records: Iterable[dict[str, Any]], *, sequence_length: int, pad_id: int = 0
) -> tuple[list[list[int]], list[int]]:
    """Generate the legacy prefix-to-next-event samples for DeepLog."""

    inputs: list[list[int]] = []
    targets: list[int] = []
    for sequence in reconstruct_group_sequences(records, pad_id=pad_id):
        for target_index in range(1, len(sequence)):
            prefix = sequence[max(0, target_index - sequence_length) : target_index]
            inputs.append([pad_id] * (sequence_length - len(prefix)) + prefix)
            targets.append(sequence[target_index])
    return inputs, targets


def build_model(config: TrainingConfig) -> Any:
    common = {
        "vocabulary_size": config.vocabulary_size,
        "sequence_length": config.sequence_length,
        "learning_rate": config.learning_rate,
    }
    if config.model == "dablog":
        return build_dablog(DablogModelConfig(**common))
    return build_deeplog(DeeplogModelConfig(**common))


def train_model(
    records: Sequence[dict[str, Any]],
    config: TrainingConfig,
    *,
    output_directory: str | Path,
    seed: int = 42,
    verbose: int = 1,
) -> dict[str, Any]:
    """Train one model and write its checkpoint, history, and metadata."""

    seed_report = set_global_seed(seed, include_tensorflow=True)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if config.model == "dablog":
        inputs, targets = dablog_examples(records, sequence_length=config.sequence_length)
    else:
        inputs, targets = deeplog_examples(records, sequence_length=config.sequence_length)

    model = build_model(config)
    import tensorflow

    tensor_inputs = tensorflow.convert_to_tensor(inputs, dtype=tensorflow.int32)
    tensor_targets = tensorflow.convert_to_tensor(targets, dtype=tensorflow.int32)
    history = model.fit(
        tensor_inputs,
        tensor_targets,
        shuffle=True,
        batch_size=config.effective_batch_size,
        epochs=config.epochs,
        callbacks=build_callbacks(output, patience=config.patience),
        verbose=verbose,
    )
    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    metadata = {
        "config": asdict(config),
        "effective_batch_size": config.effective_batch_size,
        "examples": len(inputs),
        "seed": seed,
        "seeded_libraries": asdict(seed_report),
        "python": platform.python_version(),
        "tensorflow": tensorflow.__version__,
        "history": history_data,
    }
    (output / "history.json").write_text(
        json.dumps(history_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    model.save(output / "final.keras")
    return metadata
