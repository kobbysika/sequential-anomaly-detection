"""Deterministic event vocabulary for Chapter 4 sequence models."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from sequential_anomaly.data.sequences import (
    END_TOKEN,
    PAD_TOKEN,
    START_TOKEN,
    SequenceWindow,
)

UNKNOWN_TOKEN = "<unknown>"
SPECIAL_TOKENS = (PAD_TOKEN, START_TOKEN, END_TOKEN, UNKNOWN_TOKEN)


@dataclass(frozen=True)
class EncodedWindow:
    """A textual SequenceWindow encoded as deterministic integer IDs."""

    group_id: str
    event_ids: tuple[int, ...]
    is_anomalous: bool
    attack_categories: tuple[str, ...]

    @property
    def reversed_target_ids(self) -> tuple[int, ...]:
        """Return the reverse-sequence target used by the legacy DabLog model."""

        return tuple(reversed(self.event_ids))


class EventVocabulary:
    """Bidirectional mapping between event-key strings and integer IDs."""

    def __init__(self, events: Iterable[str] = ()) -> None:
        learned_events = sorted(set(events).difference(SPECIAL_TOKENS))
        ordered_events = (*SPECIAL_TOKENS, *learned_events)
        self._id_to_event = tuple(ordered_events)
        self._event_to_id = {
            event: event_id for event_id, event in enumerate(self._id_to_event)
        }

    @classmethod
    def fit(cls, windows: Iterable[SequenceWindow]) -> EventVocabulary:
        """Fit a vocabulary from training windows only."""

        return cls(event for window in windows for event in window.events)

    def __len__(self) -> int:
        return len(self._id_to_event)

    @property
    def pad_id(self) -> int:
        return self._event_to_id[PAD_TOKEN]

    @property
    def start_id(self) -> int:
        return self._event_to_id[START_TOKEN]

    @property
    def end_id(self) -> int:
        return self._event_to_id[END_TOKEN]

    @property
    def unknown_id(self) -> int:
        return self._event_to_id[UNKNOWN_TOKEN]

    def encode_event(self, event: str) -> int:
        """Encode one event, mapping unseen values to the unknown token."""

        return self._event_to_id.get(event, self.unknown_id)

    def decode_event(self, event_id: int) -> str:
        """Decode one valid event ID."""

        if event_id < 0:
            raise ValueError(f"Unknown event ID: {event_id}")
        try:
            return self._id_to_event[event_id]
        except IndexError as error:
            raise ValueError(f"Unknown event ID: {event_id}") from error

    def encode_window(self, window: SequenceWindow) -> EncodedWindow:
        """Encode one fixed-length textual sequence window."""

        return EncodedWindow(
            group_id=window.group_id,
            event_ids=tuple(self.encode_event(event) for event in window.events),
            is_anomalous=window.is_anomalous,
            attack_categories=window.attack_categories,
        )

    def to_dict(self) -> dict[str, list[str]]:
        """Return a serializable representation with stable ID ordering."""

        return {"events": list(self._id_to_event)}

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> EventVocabulary:
        """Restore and validate a serialized vocabulary."""

        events = data.get("events")
        if not isinstance(events, list) or tuple(events[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError("Vocabulary does not contain the expected special-token prefix")

        vocabulary = cls()
        vocabulary._id_to_event = tuple(events)
        vocabulary._event_to_id = {
            event: event_id for event_id, event in enumerate(vocabulary._id_to_event)
        }
        if len(vocabulary._event_to_id) != len(vocabulary._id_to_event):
            raise ValueError("Vocabulary contains duplicate events")
        return vocabulary

    def save(self, path: str | Path) -> None:
        """Save the vocabulary as readable JSON."""

        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EventVocabulary:
        """Load a vocabulary previously written by save()."""

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
