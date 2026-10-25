import importlib.util
import unittest

from sequential_anomaly.models.deeplog import (
    DeeplogModelConfig,
    build_deeplog,
    lstm_unit_plan,
)


class DeeplogArchitectureTests(unittest.TestCase):
    def test_chapter_defaults_match_legacy_lstm_stack(self):
        config = DeeplogModelConfig(vocabulary_size=706)
        self.assertEqual(lstm_unit_plan(config), (64, 64))
        self.assertEqual(config.sequence_length, 10)

    def test_invalid_settings_are_rejected(self):
        for settings in (
            {"vocabulary_size": 4},
            {"vocabulary_size": 10, "sequence_length": 0},
            {"vocabulary_size": 10, "hidden_layers": 0},
            {"vocabulary_size": 10, "hidden_units": 0},
            {"vocabulary_size": 10, "embedding_dim": 0},
            {"vocabulary_size": 10, "learning_rate": 0},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                DeeplogModelConfig(**settings)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
    def test_tensorflow_model_shapes(self):
        model = build_deeplog(DeeplogModelConfig(vocabulary_size=100))
        self.assertEqual(model.input_shape, (None, 10))
        self.assertEqual(model.output_shape, (None, 100))
        self.assertIn("event_probabilities", {layer.name for layer in model.layers})


if __name__ == "__main__":
    unittest.main()
