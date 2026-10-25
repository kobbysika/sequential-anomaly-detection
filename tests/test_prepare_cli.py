import csv
import json
import tempfile
import unittest
from pathlib import Path

from sequential_anomaly.config import ExperimentConfig, PathConfig, PreprocessingConfig
from sequential_anomaly.data.schema import COLUMN_INDEX, EXPECTED_COLUMN_COUNT
from sequential_anomaly.data.split import build_legacy_manifest
from sequential_anomaly.cli.prepare_data import run_preprocessing


def raw_row(*, source: str, destination: str, label: str, attack: str) -> list[str]:
    row = ["0"] * EXPECTED_COLUMN_COUNT
    row[COLUMN_INDEX["srcip"]] = source
    row[COLUMN_INDEX["dstip"]] = destination
    row[COLUMN_INDEX["proto"]] = "tcp"
    row[COLUMN_INDEX["state"]] = "FIN"
    row[COLUMN_INDEX["service"]] = "http"
    row[COLUMN_INDEX["spkts"]] = "10"
    row[COLUMN_INDEX["dpkts"]] = "5"
    row[COLUMN_INDEX["stime"]] = "1421920800"
    row[COLUMN_INDEX["attack_cat"]] = attack
    row[COLUMN_INDEX["label"]] = label
    return row


class PrepareCommandTests(unittest.TestCase):
    def test_complete_synthetic_manifest_is_encoded_and_summarized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            output = root / "output"
            manifest = build_legacy_manifest(dataset_root)
            for item in manifest.all_files:
                item.path.parent.mkdir(parents=True, exist_ok=True)
                is_victim = item.role == "victim"
                row = raw_row(
                    source="59.166.0.1",
                    destination=item.ip_address,
                    label="1" if is_victim else "0",
                    attack="DoS" if is_victim else "Normal",
                )
                with item.path.open("w", encoding="utf-8", newline="") as stream:
                    csv.writer(stream).writerow(row)

            config = ExperimentConfig(
                name="synthetic",
                seed=42,
                paths=PathConfig(dataset_root, root / "results", root / "artifacts"),
                preprocessing=PreprocessingConfig(sequence_length=5),
            )
            summary = run_preprocessing(config, output_directory=output)

            saved_summary = json.loads((output / "summary.json").read_text())
            vocabulary = json.loads((output / "vocabulary.json").read_text())
            training_lines = (output / "train.jsonl").read_text().splitlines()
            normal_lines = (output / "test_normal.jsonl").read_text().splitlines()
            victim_lines = (output / "test_victim.jsonl").read_text().splitlines()

        self.assertEqual(summary, saved_summary)
        self.assertEqual(summary["partitions"]["training"]["windows"], 10)
        self.assertEqual(summary["partitions"]["testing_normal"]["windows"], 10)
        self.assertEqual(summary["partitions"]["testing_victim"]["windows"], 10)
        self.assertEqual(summary["partitions"]["testing_victim"]["anomalous"], 10)
        self.assertEqual(len(training_lines), 10)
        self.assertEqual(len(normal_lines), 10)
        self.assertEqual(len(victim_lines), 10)
        self.assertGreaterEqual(len(vocabulary["events"]), 5)


if __name__ == "__main__":
    unittest.main()
