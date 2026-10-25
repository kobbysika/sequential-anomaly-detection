import unittest

from sequential_anomaly.evaluation.scoring import (
    aggregate_group_scores,
    dablog_window_score,
    deeplog_scoring_examples,
    normalized_rank,
    score_deeplog_predictions,
)


class AnomalyScoringTests(unittest.TestCase):
    def test_normalized_rank_matches_legacy_definition(self):
        probabilities = [0.1, 0.6, 0.2, 0.1]
        self.assertEqual(normalized_rank(probabilities, 1), 0.0)
        self.assertEqual(normalized_rank(probabilities, 2), 0.25)
        self.assertEqual(normalized_rank(probabilities, 0), 0.5)

    def test_dablog_uses_worst_event_rank(self):
        probabilities = [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]
        self.assertEqual(dablog_window_score(probabilities, [1, 1]), 1 / 3)

    def test_group_aggregation_keeps_maximum_and_labels(self):
        records = [
            ({"group_id": "g", "is_anomalous": False, "attack_categories": []}, 0.2),
            ({"group_id": "g", "is_anomalous": True, "attack_categories": ["Fuzzers"]}, 0.7),
        ]
        result = aggregate_group_scores(records)[0]
        self.assertEqual(result.score, 0.7)
        self.assertTrue(result.is_anomalous)
        self.assertEqual(result.attack_categories, ("Fuzzers",))

    def test_deeplog_samples_retain_group_metadata(self):
        records = [
            {
                "group_id": "g",
                "event_ids": [0, 1, 4, 2],
                "is_anomalous": True,
                "attack_categories": ["Exploits"],
            }
        ]
        inputs, targets, metadata = deeplog_scoring_examples(records, sequence_length=4)
        self.assertEqual(inputs, [[0, 0, 0, 1], [0, 0, 1, 4]])
        self.assertEqual(targets, [4, 2])
        predictions = [[0.0, 0.0, 0.0, 0.1, 0.9], [0.0, 0.0, 0.8, 0.1, 0.1]]
        scored = score_deeplog_predictions(metadata, targets, predictions)
        self.assertEqual(len(scored), 1)
        self.assertTrue(scored[0].is_anomalous)

    def test_invalid_probability_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            normalized_rank([], 0)
        with self.assertRaises(ValueError):
            normalized_rank([1.0], 1)


if __name__ == "__main__":
    unittest.main()
