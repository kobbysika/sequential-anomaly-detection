
"""Legacy-compatible anomaly scoring for DeepLog and DabLog predictions."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from sequential_anomaly.training.train import reconstruct_group_sequences


@dataclass(frozen=True)
class AnomalyScore:
    group_id: str
    score: float
    is_anomalous: bool
    attack_categories: tuple[str, ...]


def normalized_rank(probabilities: list[float], true_event_id: int) -> float:
    """Return the supplied experiment's zero-based normalized rank score."""

    if not probabilities:
        raise ValueError("probabilities must not be empty")
    if not 0 <= true_event_id < len(probabilities):
        raise ValueError("true_event_id is outside the probability vector")
    true_probability = probabilities[true_event_id]
    rank = sum(value > true_probability for value in probabilities)
    return rank / len(probabilities)


def dablog_window_score(
    probabilities: list[list[float]], target_ids: list[int]
) -> float:
    """Use the worst reconstructed-event rank, as in the legacy DabLog report."""

    if len(probabilities) != len(target_ids) or not target_ids:
        raise ValueError("DabLog probabilities and targets must have equal nonzero length")
    return max(
        normalized_rank(event_probabilities, target_id)
        for event_probabilities, target_id in zip(probabilities, target_ids)
    )


def aggregate_group_scores(
    window_scores: list[tuple[dict[str, Any], float]],
) -> list[AnomalyScore]:
    """Aggregate overlapping samples by their maximum anomaly rank."""

    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for record, score in window_scores:
        group_id = record["group_id"]
        state = grouped.setdefault(
            group_id,
            {
                "score": 0.0,
                "is_anomalous": False,
                "attack_categories": set(),
            },
        )
        state["score"] = max(state["score"], float(score))
        state["is_anomalous"] = state["is_anomalous"] or bool(
            record.get("is_anomalous", False)
        )
        state["attack_categories"].update(record.get("attack_categories", ()))
    return [
        AnomalyScore(
            group_id=group_id,
            score=state["score"],
            is_anomalous=state["is_anomalous"],
            attack_categories=tuple(sorted(state["attack_categories"])),
        )
        for group_id, state in grouped.items()
    ]


def score_dablog_records(
    records: list[dict[str, Any]], probabilities: Any
) -> list[AnomalyScore]:
    """Score DabLog output and aggregate rolling windows into flow groups."""

    predictions = probabilities.tolist() if hasattr(probabilities, "tolist") else probabilities
    if len(records) != len(predictions):
        raise ValueError("Prediction count does not match DabLog record count")
    window_scores = []
    for record, prediction in zip(records, predictions):
        targets = record.get("reversed_target_ids", list(reversed(record["event_ids"])))
        window_scores.append((record, dablog_window_score(prediction, targets)))
    return aggregate_group_scores(window_scores)


def deeplog_scoring_examples(
    records: list[dict[str, Any]], *, sequence_length: int, pad_id: int = 0
) -> tuple[list[list[int]], list[int], list[dict[str, Any]]]:
    """Build DeepLog samples while retaining the owning group metadata."""

    metadata_by_group: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for record in records:
        metadata_by_group.setdefault(record["group_id"], record)
    sequences = reconstruct_group_sequences(records, pad_id=pad_id)
    inputs: list[list[int]] = []
    targets: list[int] = []
    metadata: list[dict[str, Any]] = []
    for record, sequence in zip(metadata_by_group.values(), sequences):
        for target_index in range(1, len(sequence)):
            prefix = sequence[max(0, target_index - sequence_length) : target_index]
            inputs.append([pad_id] * (sequence_length - len(prefix)) + prefix)
            targets.append(sequence[target_index])
            metadata.append(record)
    return inputs, targets, metadata


def score_deeplog_predictions(
    metadata: list[dict[str, Any]], targets: list[int], probabilities: Any
) -> list[AnomalyScore]:
    """Score next-event predictions and aggregate them into flow groups."""

    predictions = probabilities.tolist() if hasattr(probabilities, "tolist") else probabilities
    if not (len(metadata) == len(targets) == len(predictions)):
        raise ValueError("DeepLog prediction, target, and metadata counts must match")
    return aggregate_group_scores(
        [
            (record, normalized_rank(prediction, target))
            for record, target, prediction in zip(metadata, targets, predictions)
        ]
    )
