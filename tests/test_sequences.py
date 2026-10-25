
import unittest
from datetime import datetime, timezone

from sequential_anomaly.data.schema import COLUMN_INDEX, EXPECTED_COLUMN_COUNT
from sequential_anomaly.data.sequences import (
    END_TOKEN,
    PAD_TOKEN,
    START_TOKEN,
    EventGroup,
    build_windows,
    group_rows,
)


def flow_row(
    *,
    source: str = "59.166.0.1",
    destination: str = "149.171.126.5",
    minute: int = 0,
    protocol: str = "tcp",
    label: int = 0,
    attack: str = "Normal",
) -> list[str]:
    row = ["0"] * EXPECTED_COLUMN_COUNT
    row[COLUMN_INDEX["srcip"]] = source
    row[COLUMN_INDEX["dstip"]] = destination
    row[COLUMN_INDEX["proto"]] = protocol
    row[COLUMN_INDEX["state"]] = "FIN"
    row[COLUMN_INDEX["service"]] = "http"
    row[COLUMN_INDEX["spkts"]] = "10"
    row[COLUMN_INDEX["dpkts"]] = "5"
    row[COLUMN_INDEX["stime"]] = str(
        int(datetime(2015, 1, 22, 10, minute, tzinfo=timezone.utc).timestamp())
    )
    row[COLUMN_INDEX["attack_cat"]] = attack
    row[COLUMN_INDEX["label"]] = str(label)
    return row


class SequenceGroupingTests(unittest.TestCase):
    def test_rows_are_grouped_by_pair_and_fifteen_minute_window(self):
        groups = group_rows(
            [
                flow_row(minute=1, protocol="tcp"),
                flow_row(minute=14, protocol="udp"),
                flow_row(minute=15, protocol="icmp"),
                flow_row(minute=1, destination="149.171.126.6"),
            ]
        )

        self.assertEqual(len(groups), 3)
        self.assertEqual(len(groups[0].event_keys), 2)
        self.assertTrue(groups[0].event_keys[0].startswith("tcp,"))
        self.assertTrue(groups[0].event_keys[1].startswith("udp,"))
        self.assertTrue(groups[1].event_keys[0].startswith("icmp,"))

    def test_attack_categories_are_group_labels(self):
        groups = group_rows(
            [
                flow_row(minute=1),
                flow_row(minute=2, label=1, attack="Exploits"),
                flow_row(minute=3, label=1, attack="Exploits"),
            ]
        )

        self.assertTrue(groups[0].is_anomalous)
        self.assertEqual(groups[0].attack_categories, ("Exploits",))

    def test_short_group_is_left_padded_and_bounded(self):
        group = EventGroup("group", ("A",), ())
        windows = build_windows(group, sequence_length=5)

        self.assertEqual(len(windows), 1)
        self.assertEqual(
            windows[0].events,
            (PAD_TOKEN, PAD_TOKEN, START_TOKEN, "A", END_TOKEN),
        )
        self.assertFalse(windows[0].is_anomalous)

    def test_long_group_produces_rolling_windows(self):
        group = EventGroup("group", ("A", "B", "C", "D", "E"), ("DoS",))
        windows = build_windows(group, sequence_length=4)

        self.assertEqual(
            [window.events for window in windows],
            [
                (START_TOKEN, "A", "B", "C"),
                ("A", "B", "C", "D"),
                ("B", "C", "D", "E"),
                ("C", "D", "E", END_TOKEN),
            ],
        )
        self.assertTrue(all(window.is_anomalous for window in windows))
        self.assertTrue(all(window.attack_categories == ("DoS",) for window in windows))

    def test_invalid_sequence_length_is_rejected(self):
        with self.assertRaises(ValueError):
            build_windows(EventGroup("group", (), ()), sequence_length=0)


if __name__ == "__main__":
    unittest.main()
