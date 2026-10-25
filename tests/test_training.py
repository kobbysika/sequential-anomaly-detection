import json
import tempfile
import unittest
from pathlib import Path

from sequential_anomaly.training.train import (
    TrainingConfig,
    dablog_examples,
    deeplog_examples,
    read_prepared_windows,
    reconstruct_group_sequences,
)


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"group_id": "a", "event_ids": [0, 1, 4, 5], "reversed_target_ids": [5, 4, 1, 0]},
            {"group_id": "b", "event_ids": [1, 6, 7, 8], "reversed_target_ids": [8, 7, 6, 1]},
            {"group_id": "b", "event_ids": [6, 7, 8, 2], "reversed_target_ids": [2, 8, 7, 6]},
        ]

    def test_reconstructs_groups_from_overlapping_windows(self):
        self.assertEqual(reconstruct_group_sequences(self.records), [[1, 4, 5], [1, 6, 7, 8, 2]])

    def test_deeplog_creates_padded_prefix_next_event_examples(self):
        inputs, targets = deeplog_examples(self.records, sequence_length=4)
        self.assertEqual(inputs[0], [0, 0, 0, 1])
        self.assertEqual(targets[0], 4)
        self.assertEqual(inputs[-1], [1, 6, 7, 8])
        self.assertEqual(targets[-1], 2)

    def test_dablog_uses_reverse_sequence_targets(self):
        inputs, targets = dablog_examples(self.records, sequence_length=4)
        self.assertEqual(inputs[0], [0, 1, 4, 5])
        self.assertEqual(targets[0], [5, 4, 1, 0])

    def test_legacy_batch_defaults_remain_model_specific(self):
        self.assertEqual(TrainingConfig("dablog", 10).effective_batch_size, 256)
        self.assertEqual(TrainingConfig("deeplog", 10).effective_batch_size, 2048)

    def test_jsonl_reader_rejects_empty_and_accepts_valid_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(self.records[0]) + "\n", encoding="utf-8")
            self.assertEqual(read_prepared_windows(path), [self.records[0]])
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_prepared_windows(path)


if __name__ == "__main__":
    unittest.main()
