
"""CSV validation and preprocessing orchestration for UNSW-NB15."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from sequential_anomaly.data.schema import COLUMN_INDEX, validate_row_width
from sequential_anomaly.data.sequences import (
    SequenceWindow,
    build_windows_for_groups,
    group_rows,
)
from sequential_anomaly.data.split import DatasetFile, LegacyDatasetManifest

REQUIRED_INTEGER_FIELDS = ("spkts", "dpkts", "stime")
OPTIONAL_INTEGER_FIELDS = (
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "trans_depth",
)


class UNSWDataError(ValueError):
    """A malformed UNSW-NB15 record with source location context."""


@dataclass(frozen=True)
class PreparedLegacyDataset:
    """Sequence windows produced from the original Chapter 4 file selection."""

    training: tuple[SequenceWindow, ...]
    testing_normal: tuple[SequenceWindow, ...]
    testing_victim: tuple[SequenceWindow, ...]

    @property
    def testing(self) -> tuple[SequenceWindow, ...]:
        return self.testing_normal + self.testing_victim


def validate_row(row: Sequence[str]) -> None:
    """Validate fields required by the faithful event and sequence pipeline."""

    validate_row_width(list(row))

    for field in REQUIRED_INTEGER_FIELDS:
        value = row[COLUMN_INDEX[field]].strip()
        if not value:
            raise ValueError(f"{field} must not be blank")
        try:
            int(value)
        except ValueError as error:
            raise ValueError(f"{field} must be an integer, got {value!r}") from error

    for field in OPTIONAL_INTEGER_FIELDS:
        value = row[COLUMN_INDEX[field]].strip()
        if not value:
            continue
        try:
            int(value)
        except ValueError as error:
            raise ValueError(f"{field} must be blank or an integer, got {value!r}") from error

    label = row[COLUMN_INDEX["label"]].strip()
    if label not in {"0", "1"}:
        raise ValueError(f"label must be '0' or '1', got {label!r}")
    if label == "1" and not row[COLUMN_INDEX["attack_cat"]].strip():
        raise ValueError("attack_cat must not be blank when label is '1'")


def read_unsw_rows(path: str | Path) -> Iterator[list[str]]:
    """Stream validated records from one headerless UNSW-NB15 CSV file."""

    source = Path(path)
    try:
        stream = source.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise UNSWDataError(f"Cannot open {source}: {error}") from error

    with stream:
        reader = csv.reader(stream)
        try:
            for line_number, row in enumerate(reader, start=1):
                if not row or all(not value.strip() for value in row):
                    continue
                try:
                    validate_row(row)
                except ValueError as error:
                    raise UNSWDataError(f"{source}:{line_number}: {error}") from error
                yield row
        except csv.Error as error:
            raise UNSWDataError(
                f"{source}:{reader.line_num}: invalid CSV: {error}"
            ) from error


def prepare_files(
    files: Iterable[DatasetFile],
    *,
    window_minutes: int = 15,
    sequence_length: int = 10,
    keyset: int = 0,
    packet_divisor: int = 100,
) -> tuple[SequenceWindow, ...]:
    """Read directional files and convert their rows into model windows."""

    windows: list[SequenceWindow] = []
    for dataset_file in files:
        groups = group_rows(
            read_unsw_rows(dataset_file.path),
            window_minutes=window_minutes,
            keyset=keyset,
            packet_divisor=packet_divisor,
        )
        windows.extend(
            build_windows_for_groups(groups, sequence_length=sequence_length)
        )
    return tuple(windows)


def prepare_legacy_dataset(
    manifest: LegacyDatasetManifest,
    *,
    window_minutes: int = 15,
    sequence_length: int = 10,
    keyset: int = 0,
    packet_divisor: int = 100,
) -> PreparedLegacyDataset:
    """Prepare the original training, normal-test, and victim-test partitions."""

    options = {
        "window_minutes": window_minutes,
        "sequence_length": sequence_length,
        "keyset": keyset,
        "packet_divisor": packet_divisor,
    }
    return PreparedLegacyDataset(
        training=prepare_files(manifest.training, **options),
        testing_normal=prepare_files(manifest.testing_normal, **options),
        testing_victim=prepare_files(manifest.testing_victim, **options),
    )
