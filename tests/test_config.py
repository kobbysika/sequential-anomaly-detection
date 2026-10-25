import json
import tempfile
import unittest
from pathlib import Path

from sequential_anomaly.config import (
    ExperimentConfig,
    PathConfig,
    PreprocessingConfig,
    load_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_shipped_configurations_load(self):
        for name in ("pilot", "deeplog", "dablog"):
            config = load_config(PROJECT_ROOT / "configs" / f"{name}.yaml")
            self.assertEqual(config.name, name)
            self.assertEqual(config.preprocessing.sequence_length, 10)
            self.assertEqual(config.preprocessing.window_minutes, 15)
            self.assertEqual(config.preprocessing.keyset, 0)
            self.assertEqual(config.preprocessing.packet_divisor, 100)

    def test_path_values_remain_portable_relative_paths(self):
        config = load_config(PROJECT_ROOT / "configs" / "pilot.yaml")

        self.assertFalse(config.paths.dataset_root.is_absolute())
        self.assertFalse(config.paths.results_root.is_absolute())
        self.assertFalse(config.paths.artifacts_root.is_absolute())

    def test_preprocessing_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            PreprocessingConfig(window_minutes=7)
        with self.assertRaises(ValueError):
            PreprocessingConfig(sequence_length=0)
        with self.assertRaises(ValueError):
            PreprocessingConfig(keyset=5)
        with self.assertRaises(ValueError):
            PreprocessingConfig(packet_divisor=0)

    def test_experiment_rejects_blank_name_and_negative_seed(self):
        paths = PathConfig(Path("data"), Path("results"), Path("artifacts"))
        preprocessing = PreprocessingConfig()
        with self.assertRaises(ValueError):
            ExperimentConfig(" ", 42, paths, preprocessing)
        with self.assertRaises(ValueError):
            ExperimentConfig("test", -1, paths, preprocessing)

    def test_invalid_file_has_actionable_location(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("{bad", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"bad\.yaml:1:2"):
                load_config(path)

    def test_missing_required_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.yaml"
            path.write_text(json.dumps({"name": "test"}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Missing configuration settings"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
