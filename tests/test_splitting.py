
import tempfile
import unittest
from pathlib import Path

from sequential_anomaly.data.split import (
    ATTACKER_IPS,
    NORMAL_DESTINATION_IPS,
    NORMAL_SENDER_IPS,
    VICTIM_IPS,
    build_legacy_manifest,
)


class LegacySplitTests(unittest.TestCase):
    def test_address_roles_match_the_supplied_experiment(self):
        self.assertEqual(NORMAL_SENDER_IPS[0], "59.166.0.0")
        self.assertEqual(NORMAL_SENDER_IPS[-1], "59.166.0.9")
        self.assertEqual(NORMAL_DESTINATION_IPS[0], "149.171.126.0")
        self.assertEqual(NORMAL_DESTINATION_IPS[-1], "149.171.126.9")
        self.assertEqual(VICTIM_IPS[0], "149.171.126.10")
        self.assertEqual(VICTIM_IPS[-1], "149.171.126.19")
        self.assertEqual(ATTACKER_IPS, tuple(f"175.45.176.{index}" for index in range(4)))

    def test_role_address_sets_do_not_overlap(self):
        role_sets = [
            set(NORMAL_SENDER_IPS),
            set(NORMAL_DESTINATION_IPS),
            set(VICTIM_IPS),
            set(ATTACKER_IPS),
        ]
        for index, addresses in enumerate(role_sets):
            for other in role_sets[index + 1 :]:
                self.assertTrue(addresses.isdisjoint(other))

    def test_manifest_uses_original_directional_filenames(self):
        root = Path("datasets/processed/unsw-nb15")
        manifest = build_legacy_manifest(root)

        self.assertEqual(len(manifest.training), 10)
        self.assertEqual(len(manifest.testing_normal), 10)
        self.assertEqual(len(manifest.testing_victim), 10)
        self.assertTrue(
            all(item.path.name == "flow_from_this.csv" for item in manifest.training)
        )
        self.assertTrue(
            all(item.path.name == "flow_to_this.csv" for item in manifest.testing)
        )
        self.assertTrue(all(item.split == "train" for item in manifest.training))
        self.assertTrue(all(item.split == "test" for item in manifest.testing))

    def test_attacker_addresses_are_not_direct_input_files(self):
        manifest = build_legacy_manifest("dataset")
        selected_addresses = {item.ip_address for item in manifest.all_files}

        self.assertTrue(set(ATTACKER_IPS).isdisjoint(selected_addresses))

    def test_required_file_check_accepts_complete_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_legacy_manifest(directory)
            for item in manifest.all_files:
                item.path.parent.mkdir(parents=True, exist_ok=True)
                item.path.touch()

            checked = build_legacy_manifest(directory, require_files=True)

        self.assertEqual(len(checked.all_files), 30)

    def test_required_file_check_reports_missing_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Missing 30"):
                build_legacy_manifest(directory, require_files=True)


if __name__ == "__main__":
    unittest.main()
