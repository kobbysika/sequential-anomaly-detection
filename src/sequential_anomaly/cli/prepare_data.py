
"""Command-line entry point for the Chapter 4 preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from sequential_anomaly.config import ExperimentConfig, load_config
from sequential_anomaly.data.split import build_legacy_manifest
from sequential_anomaly.data.validate import prepare_legacy_dataset
from sequential_anomaly.data.vocabulary import EncodedWindow, EventVocabulary
from sequential_anomaly.reproducibility import set_global_seed


def _encoded_record(window: EncodedWindow) -> dict[str, object]:
    return {
        "group_id": window.group_id,
        "event_ids": list(window.event_ids),
        "reversed_target_ids": list(window.reversed_target_ids),
        "is_anomalous": window.is_anomalous,
        "attack_categories": list(window.attack_categories),
    }


def _write_jsonl(path: Path, windows: Iterable[EncodedWindow]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for window in windows:
            stream.write(json.dumps(_encoded_record(window), sort_keys=True) + "\n")
            count += 1
    return count


def _partition_summary(windows: tuple[EncodedWindow, ...]) -> dict[str, int]:
    anomalous = sum(window.is_anomalous for window in windows)
    return {
        "windows": len(windows),
        "normal": len(windows) - anomalous,
        "anomalous": anomalous,
    }


def run_preprocessing(
    config: ExperimentConfig, *, output_directory: str | Path | None = None
) -> dict[str, object]:
    """Run the configured legacy-compatible preprocessing pipeline."""

    seed_report = set_global_seed(config.seed, include_tensorflow=False)
    manifest = build_legacy_manifest(config.paths.dataset_root, require_files=True)
    options = {
        "window_minutes": config.preprocessing.window_minutes,
        "sequence_length": config.preprocessing.sequence_length,
        "keyset": config.preprocessing.keyset,
        "packet_divisor": config.preprocessing.packet_divisor,
    }
    prepared = prepare_legacy_dataset(manifest, **options)
    vocabulary = EventVocabulary.fit(prepared.training)

    encoded_training = tuple(vocabulary.encode_window(window) for window in prepared.training)
    encoded_normal = tuple(
        vocabulary.encode_window(window) for window in prepared.testing_normal
    )
    encoded_victim = tuple(
        vocabulary.encode_window(window) for window in prepared.testing_victim
    )

    output = Path(output_directory) if output_directory else config.paths.results_root / "preprocessed"
    output.mkdir(parents=True, exist_ok=True)
    vocabulary.save(output / "vocabulary.json")
    _write_jsonl(output / "train.jsonl", encoded_training)
    _write_jsonl(output / "test_normal.jsonl", encoded_normal)
    _write_jsonl(output / "test_victim.jsonl", encoded_victim)

    summary: dict[str, object] = {
        "experiment": config.name,
        "seed": config.seed,
        "seeded_libraries": {
            "python": seed_report.python_seeded,
            "numpy": seed_report.numpy_seeded,
            "tensorflow": seed_report.tensorflow_seeded,
        },
        "preprocessing": {
            "window_minutes": config.preprocessing.window_minutes,
            "sequence_length": config.preprocessing.sequence_length,
            "keyset": config.preprocessing.keyset,
            "packet_divisor": config.preprocessing.packet_divisor,
        },
        "vocabulary_size": len(vocabulary),
        "partitions": {
            "training": _partition_summary(encoded_training),
            "testing_normal": _partition_summary(encoded_normal),
            "testing_victim": _partition_summary(encoded_victim),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Chapter 4 UNSW-NB15 sequence data"
    )
    parser.add_argument("--config", required=True, help="Path to a project configuration")
    parser.add_argument(
        "--output",
        help="Optional output directory override (defaults to results_root/preprocessed)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_preprocessing(load_config(args.config), output_directory=args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
