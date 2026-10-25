import importlib.util
import unittest

from sequential_anomaly.models.dablog import (
    DablogModelConfig,
    build_dablog,
    lstm_unit_plan,
)


class DablogArchitectureTests(unittest.TestCase):
    def test_chapter_defaults_create_mirrored_lstm_widths(self):
        config = DablogModelConfig(vocabulary_size=706)

        self.assertEqual(lstm_unit_plan(config), ((64, 32), (32, 64)))
        self.assertEqual(config.sequence_length, 10)
        self.assertEqual(config.embedding_dim, 64)

    def test_deeper_plan_follows_legacy_integer_division(self):
        config = DablogModelConfig(
            vocabulary_size=100, hidden_layers=3, hidden_units=60
        )

        self.assertEqual(lstm_unit_plan(config), ((60, 30, 20), (20, 30, 60)))

    def test_invalid_model_settings_are_rejected(self):
        invalid = (
            {"vocabulary_size": 4},
            {"vocabulary_size": 10, "sequence_length": 0},
            {"vocabulary_size": 10, "hidden_layers": 0},
            {"vocabulary_size": 10, "hidden_units": 0},
            {"vocabulary_size": 10, "embedding_dim": 0},
            {"vocabulary_size": 10, "learning_rate": 0},
        )
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                DablogModelConfig(**settings)

    @unittest.skipUnless(
        importlib.util.find_spec("tensorflow"), "TensorFlow is not installed"
    )
    def test_tensorflow_model_has_expected_input_and_output_shapes(self):
        model = build_dablog(
            DablogModelConfig(vocabulary_size=100, sequence_length=10)
        )

        self.assertEqual(model.input_shape, (None, 10))
        self.assertEqual(model.output_shape, (None, 10, 100))
        layer_names = {layer.name for layer in model.layers}
        self.assertIn("event_embedding", layer_names)
        self.assertIn("repeat_bottleneck", layer_names)
        self.assertIn("event_probabilities", layer_names)


if __name__ == "__main__":
    unittest.main()
