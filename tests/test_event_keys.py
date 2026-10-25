
import unittest

from sequential_anomaly.data.event_keys import attack_category, build_event_key
from sequential_anomaly.data.schema import COLUMN_INDEX, EXPECTED_COLUMN_COUNT


def sample_row() -> list[str]:
    row = ["0"] * EXPECTED_COLUMN_COUNT
    row[COLUMN_INDEX["srcip"]] = "59.166.0.1"
    row[COLUMN_INDEX["dstip"]] = "149.171.126.5"
    row[COLUMN_INDEX["proto"]] = "tcp"
    row[COLUMN_INDEX["state"]] = "FIN"
    row[COLUMN_INDEX["service"]] = "http"
    row[COLUMN_INDEX["spkts"]] = "140"
    row[COLUMN_INDEX["dpkts"]] = "75"
    row[COLUMN_INDEX["ct_flw_http_mthd"]] = "1"
    row[COLUMN_INDEX["attack_cat"]] = "Normal"
    row[COLUMN_INDEX["label"]] = "0"
    return row


class EventKeyTests(unittest.TestCase):
    def test_default_key_matches_legacy_field_order(self):
        self.assertEqual(
            build_event_key(sample_row()),
            "tcp,FIN,http,0,0,1,0,0,2",
        )

    def test_keysets_progressively_remove_fields(self):
        row = sample_row()
        self.assertEqual(build_event_key(row, keyset=1), "tcp,FIN,http,0,0,1,0,0")
        self.assertEqual(build_event_key(row, keyset=2), "tcp,FIN,http,0,0,1,0")
        self.assertEqual(build_event_key(row, keyset=3), "tcp,FIN,http,0,0")
        self.assertEqual(build_event_key(row, keyset=4), "tcp,FIN,http,0")

    def test_blank_legacy_count_fields_become_zero(self):
        row = sample_row()
        for field in (
            "is_ftp_login",
            "ct_ftp_cmd",
            "ct_flw_http_mthd",
            "trans_depth",
        ):
            row[COLUMN_INDEX[field]] = " "
        self.assertEqual(build_event_key(row), "tcp,FIN,http,0,0,0,0,0,2")

    def test_input_row_is_not_modified(self):
        row = sample_row()
        original = row.copy()
        build_event_key(row)
        self.assertEqual(row, original)

    def test_attack_category(self):
        normal = sample_row()
        self.assertIsNone(attack_category(normal))

        attack = sample_row()
        attack[COLUMN_INDEX["label"]] = "1"
        attack[COLUMN_INDEX["attack_cat"]] = "Exploits"
        self.assertEqual(attack_category(attack), "Exploits")

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            build_event_key(sample_row(), keyset=5)
        with self.assertRaises(ValueError):
            build_event_key(sample_row(), packet_divisor=0)
        with self.assertRaises(ValueError):
            build_event_key(["too", "short"])


if __name__ == "__main__":
    unittest.main()
