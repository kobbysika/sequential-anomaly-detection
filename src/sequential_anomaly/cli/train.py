
"""Command-line training entry point for DabLog and DeepLog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sequential_anomaly.config import load_config
from sequential_anomaly.data.vocabulary import EventVocabulary
from sequential_anomaly.training.train import (
    TrainingConfig,
    read_prepared_windows,
    train_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Chapter 4 sequence model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=("dablog", "deeplog"))
    parser.add_argument("--prepared", help="Directory produced by prepare_data")
    parser.add_argument("--output", help="Checkpoint/output directory override")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--patience", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment = load_config(args.config)
    model_name = args.model or experiment.name
    if model_name not in {"dablog", "deeplog"}:
        raise ValueError("Pilot config requires an explicit --model")

    prepared = Path(args.prepared) if args.prepared else experiment.paths.results_root / "preprocessed"
    output = Path(args.output) if args.output else experiment.paths.artifacts_root
    vocabulary = EventVocabulary.load(prepared / "vocabulary.json")
    records = read_prepared_windows(prepared / "train.jsonl")
    training = TrainingConfig(
        model=model_name,
        vocabulary_size=len(vocabulary),
        sequence_length=experiment.preprocessing.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    metadata = train_model(
        records,
        training,
        output_directory=output,
        seed=experiment.seed,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
