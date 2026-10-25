
import unittest

from sequential_anomaly.evaluation.metrics import (
    confusion_at_threshold,
    evaluate_scores,
)
from sequential_anomaly.evaluation.scoring import AnomalyScore


def score(group_id, value, anomalous):
    return AnomalyScore(group_id, value, anomalous, ())


class EvaluationMetricTests(unittest.TestCase):
    def setUp(self):
        self.perfect = [
            score("attack-1", 0.9, True),
            score("attack-2", 0.8, True),
            score("normal-1", 0.2, False),
            score("normal-2", 0.1, False),
        ]

    def test_confusion_matrix_uses_strict_legacy_threshold(self):
        result = confusion_at_threshold(self.perfect, 0.2)
        self.assertEqual(
            (
                result.true_positive,
                result.false_positive,
                result.true_negative,
                result.false_negative,
            ),
            (2, 0, 2, 0),
        )
        self.assertEqual(result.f1, 1.0)

    def test_perfect_ranking_has_unit_auc(self):
        result = evaluate_scores(self.perfect, threshold=0.2)
        self.assertAlmostEqual(result["auroc"], 1.0)
        self.assertAlmostEqual(result["auprc"], 1.0)
        self.assertEqual(result["best_f1"], 1.0)

    def test_empty_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_scores([])


if __name__ == "__main__":
    unittest.main()
