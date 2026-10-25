import csv
import tempfile
import unittest
from pathlib import Path

from sequential_anomaly.data.schema import COLUMN_INDEX, EXPECTED_COLUMN_COUNT
from sequential_anomaly.data.split import DatasetFile, LegacyDatasetManifest
from sequential_anomaly.data.validate import (
    UNSWDataError,
    prepare_legacy_dataset,
    read_unsw_rows,
)


def valid_row(*, label: str = "0", attack: str = "Normal") -> list[str]:
    row = ["0"] * EXPECTED_COLUMN_COUNT
    row[COLUMN_INDEX["srcip"]] = "59.166.0.1"
    row[COLUMN_INDEX["dstip"]] = "149.171.126.5"
    row[COLUMN_INDEX["proto"]] = "tcp"
    row[COLUMN_INDEX["state"]] = "FIN"
    row[COLUMN_INDEX["service"]] = "http"
    row[COLUMN_INDEX["spkts"]] = "10"
    row[COLUMN_INDEX["dpkts"]] = "5"
    row[COLUMN_INDEX["stime"]] = "1421920800"
    row[COLUMN_INDEX["attack_cat"]] = attack
    row[COLUMN_INDEX["label"]] = label
    return row


def write_rows(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


class CSVValidationTests(unittest.TestCase):
    def test_reader_streams_valid_headerless_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.csv"
            write_rows(path, [valid_row(), valid_row(label="1", attack="Exploits")])

            rows = list(read_unsw_rows(path))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][COLUMN_INDEX["attack_cat"]], "Exploits")

    def test_reader_reports_file_and_line_for_bad_width(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.csv"
            write_rows(path, [valid_row(), ["too", "short"]])

            with self.assertRaisesRegex(UNSWDataError, r"flows\.csv:2: Expected 49"):
                list(read_unsw_rows(path))

    def test_reader_rejects_invalid_required_values(self):
        row = valid_row()
        row[COLUMN_INDEX["stime"]] = "not-a-timestamp"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.csv"
            write_rows(path, [row])

            with self.assertRaisesRegex(UNSWDataError, "stime must be an integer"):
                list(read_unsw_rows(path))

    def test_reader_rejects_attack_without_category(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.csv"
            write_rows(path, [valid_row(label="1", attack="")])

            with self.assertRaisesRegex(UNSWDataError, "attack_cat must not be blank"):
                list(read_unsw_rows(path))

    def test_manifest_is_connected_to_sequence_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.csv"
            normal_path = root / "normal.csv"
            victim_path = root / "victim.csv"
            write_rows(train_path, [valid_row()])
            write_rows(normal_path, [valid_row()])
            write_rows(victim_path, [valid_row(label="1", attack="DoS")])
            manifest = LegacyDatasetManifest(
                training=(DatasetFile("train", "normal_sender", "sender", train_path),),
                testing_normal=(
                    DatasetFile("test", "normal_destination", "normal", normal_path),
                ),
                testing_victim=(DatasetFile("test", "victim", "victim", victim_path),),
            )

            prepared = prepare_legacy_dataset(manifest, sequence_length=5)

        self.assertEqual(len(prepared.training), 1)
        self.assertEqual(len(prepared.testing_normal), 1)
        self.assertEqual(len(prepared.testing_victim), 1)
        self.assertFalse(prepared.training[0].is_anomalous)
        self.assertFalse(prepared.testing_normal[0].is_anomalous)
        self.assertTrue(prepared.testing_victim[0].is_anomalous)
        self.assertEqual(prepared.testing_victim[0].attack_categories, ("DoS",))


if __name__ == "__main__":
    unittest.main()
