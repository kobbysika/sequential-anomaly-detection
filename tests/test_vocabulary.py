import tempfile
import unittest
from pathlib import Path

from sequential_anomaly.data.sequences import (
    END_TOKEN,
    PAD_TOKEN,
    START_TOKEN,
    SequenceWindow,
)
from sequential_anomaly.data.vocabulary import (
    SPECIAL_TOKENS,
    UNKNOWN_TOKEN,
    EventVocabulary,
)


def window(*events: str) -> SequenceWindow:
    return SequenceWindow(
        group_id="group",
        events=tuple(events),
        is_anomalous=False,
        attack_categories=(),
    )


class EventVocabularyTests(unittest.TestCase):
    def test_special_tokens_keep_the_legacy_id_order(self):
        vocabulary = EventVocabulary.fit([window(START_TOKEN, "z", "a", END_TOKEN)])

        self.assertEqual(vocabulary.pad_id, 0)
        self.assertEqual(vocabulary.start_id, 1)
        self.assertEqual(vocabulary.end_id, 2)
        self.assertEqual(vocabulary.unknown_id, 3)
        self.assertEqual(
            [vocabulary.decode_event(index) for index in range(4)],
            list(SPECIAL_TOKENS),
        )

    def test_learned_events_are_sorted_for_reproducible_ids(self):
        vocabulary = EventVocabulary.fit([window("z", "a", "m", "a")])

        self.assertEqual(vocabulary.encode_event("a"), 4)
        self.assertEqual(vocabulary.encode_event("m"), 5)
        self.assertEqual(vocabulary.encode_event("z"), 6)

    def test_unseen_events_map_to_unknown(self):
        vocabulary = EventVocabulary.fit([window("known")])

        self.assertEqual(vocabulary.encode_event("unseen"), vocabulary.unknown_id)
        self.assertEqual(vocabulary.decode_event(vocabulary.unknown_id), UNKNOWN_TOKEN)

    def test_window_encoding_preserves_metadata_and_length(self):
        source = SequenceWindow(
            group_id="subject-1",
            events=(PAD_TOKEN, START_TOKEN, "A", END_TOKEN),
            is_anomalous=True,
            attack_categories=("Exploits",),
        )
        encoded = EventVocabulary.fit([source]).encode_window(source)

        self.assertEqual(len(encoded.event_ids), len(source.events))
        self.assertEqual(encoded.group_id, source.group_id)
        self.assertTrue(encoded.is_anomalous)
        self.assertEqual(encoded.attack_categories, ("Exploits",))

    def test_dablog_target_reverses_the_encoded_window(self):
        vocabulary = EventVocabulary.fit([window("A", "B", "C")])
        encoded = vocabulary.encode_window(window("A", "B", "C"))

        self.assertEqual(encoded.reversed_target_ids, tuple(reversed(encoded.event_ids)))

    def test_vocabulary_json_round_trip_preserves_ids(self):
        vocabulary = EventVocabulary.fit([window("B", "A")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocabulary.json"
            vocabulary.save(path)
            restored = EventVocabulary.load(path)

        self.assertEqual(restored.to_dict(), vocabulary.to_dict())
        self.assertEqual(restored.encode_event("A"), vocabulary.encode_event("A"))

    def test_invalid_serialized_vocabulary_is_rejected(self):
        with self.assertRaises(ValueError):
            EventVocabulary.from_dict({"events": ["A", "B"]})


if __name__ == "__main__":
    unittest.main()
