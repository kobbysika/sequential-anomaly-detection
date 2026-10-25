
"""Sequence grouping and window construction for the Chapter 4 replication."""

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from sequential_anomaly.data.event_keys import attack_category, build_event_key
from sequential_anomaly.data.schema import COLUMN_INDEX, validate_row_width

PAD_TOKEN = "<pad>"
START_TOKEN = "<sequence>"
END_TOKEN = "</sequence>"


@dataclass(frozen=True)
class EventGroup:
    """Events sharing a source, destination, and fixed clock-time window."""

    group_id: str
    event_keys: tuple[str, ...]
    attack_categories: tuple[str, ...]

    @property
    def is_anomalous(self) -> bool:
        return bool(self.attack_categories)


@dataclass(frozen=True)
class SequenceWindow:
    """One fixed-length model input produced from an EventGroup."""

    group_id: str
    events: tuple[str, ...]
    is_anomalous: bool
    attack_categories: tuple[str, ...]


def group_id_from_row(row: Sequence[str], *, window_minutes: int = 15) -> str:
    """Reproduce the legacy source/destination/day/hour/window grouping in UTC."""

    validate_row_width(list(row))
    if window_minutes <= 0 or 60 % window_minutes != 0:
        raise ValueError("window_minutes must be a positive divisor of 60")

    timestamp = int(row[COLUMN_INDEX["stime"]])
    moment = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    source = row[COLUMN_INDEX["srcip"]]
    destination = row[COLUMN_INDEX["dstip"]]
    minute_window = moment.minute // window_minutes
    return "-".join(
        (
            "from",
            source,
            "to",
            destination,
            "on",
            str(moment.day),
            str(moment.hour),
            str(minute_window),
        )
    )


def group_rows(
    rows: Iterable[Sequence[str]],
    *,
    window_minutes: int = 15,
    keyset: int = 0,
    packet_divisor: int = 100,
) -> list[EventGroup]:
    """Convert raw rows into ordered Chapter 4 event groups."""

    grouped: OrderedDict[str, dict[str, list[str]]] = OrderedDict()
    for row in rows:
        group_id = group_id_from_row(row, window_minutes=window_minutes)
        if group_id not in grouped:
            grouped[group_id] = {"event_keys": [], "attack_categories": []}

        grouped[group_id]["event_keys"].append(
            build_event_key(row, keyset=keyset, packet_divisor=packet_divisor)
        )
        category = attack_category(row)
        if category is not None:
            grouped[group_id]["attack_categories"].append(category)

    return [
        EventGroup(
            group_id=group_id,
            event_keys=tuple(values["event_keys"]),
            attack_categories=tuple(sorted(set(values["attack_categories"]))),
        )
        for group_id, values in grouped.items()
    ]


def build_windows(group: EventGroup, *, sequence_length: int = 10) -> list[SequenceWindow]:
    """Add boundary tokens and create the legacy rolling reconstruction windows."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than zero")

    augmented = (START_TOKEN, *group.event_keys, END_TOKEN)
    if len(augmented) <= sequence_length:
        event_windows = [
            (PAD_TOKEN,) * (sequence_length - len(augmented)) + augmented
        ]
    else:
        event_windows = [
            augmented[start : start + sequence_length]
            for start in range(len(augmented) - sequence_length + 1)
        ]

    return [
        SequenceWindow(
            group_id=group.group_id,
            events=tuple(events),
            is_anomalous=group.is_anomalous,
            attack_categories=group.attack_categories,
        )
        for events in event_windows
    ]


def build_windows_for_groups(
    groups: Iterable[EventGroup], *, sequence_length: int = 10
) -> list[SequenceWindow]:
    """Create model windows for a collection of event groups."""

    return [
        window
        for group in groups
        for window in build_windows(group, sequence_length=sequence_length)
    ]
