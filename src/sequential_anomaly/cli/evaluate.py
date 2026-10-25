
"""Evaluate a trained DabLog or DeepLog checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from sequential_anomaly.config import load_config
from sequential_anomaly.evaluation.metrics import evaluate_scores
from sequential_anomaly.evaluation.plots import save_evaluation_plots
from sequential_anomaly.evaluation.scoring import (
    deeplog_scoring_examples,
    score_dablog_records,
    score_deeplog_predictions,
)
from sequential_anomaly.training.train import read_prepared_windows


def _namespace(records: list[dict[str, object]], partition: str) -> list[dict[str, object]]:
    return [{**record, "group_id": f"{partition}:{record['group_id']}"} for record in records]


def evaluate_checkpoint(
    *,
    model_name: str,
    checkpoint: str | Path,
    prepared_directory: str | Path,
    output_directory: str | Path,
    sequence_length: int,
    threshold: float = 0.05,
    batch_size: int = 2048,
) -> dict[str, object]:
    """Run inference, aggregate flow-group scores, and save report artifacts."""

    if model_name not in {"dablog", "deeplog"}:
        raise ValueError("model_name must be 'dablog' or 'deeplog'")
    import tensorflow

    prepared = Path(prepared_directory)
    records = _namespace(read_prepared_windows(prepared / "test_normal.jsonl"), "normal")
    records += _namespace(read_prepared_windows(prepared / "test_victim.jsonl"), "victim")
    model = tensorflow.keras.models.load_model(checkpoint)

    if model_name == "dablog":
        inputs = tensorflow.convert_to_tensor(
            [record["event_ids"] for record in records], dtype=tensorflow.int32
        )
        probabilities = model.predict(inputs, batch_size=batch_size, verbose=0)
        scores = score_dablog_records(records, probabilities)
    else:
        inputs, targets, metadata = deeplog_scoring_examples(
            records, sequence_length=sequence_length
        )
        probabilities = model.predict(
            tensorflow.convert_to_tensor(inputs, dtype=tensorflow.int32),
            batch_size=batch_size,
            verbose=0,
        )
        scores = score_deeplog_predictions(metadata, targets, probabilities)

    metrics = evaluate_scores(scores, threshold=threshold)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "scores.jsonl").open("w", encoding="utf-8") as stream:
        for score in scores:
            stream.write(json.dumps(asdict(score), sort_keys=True) + "\n")
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics["curve"][0]))
        writer.writeheader()
        writer.writerows(metrics["curve"])
    save_evaluation_plots(metrics, output)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Chapter 4 sequence model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=("dablog", "deeplog"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--prepared")
    parser.add_argument("--output")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment = load_config(args.config)
    model_name = args.model or experiment.name
    if model_name not in {"dablog", "deeplog"}:
        raise ValueError("Pilot config requires an explicit --model")
    prepared = Path(args.prepared) if args.prepared else experiment.paths.results_root / "preprocessed"
    checkpoint = Path(args.checkpoint) if args.checkpoint else experiment.paths.artifacts_root / "best.keras"
    output = Path(args.output) if args.output else experiment.paths.results_root / "evaluation"
    metrics = evaluate_checkpoint(
        model_name=model_name,
        checkpoint=checkpoint,
        prepared_directory=prepared,
        output_directory=output,
        sequence_length=experiment.preprocessing.sequence_length,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "curve"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
