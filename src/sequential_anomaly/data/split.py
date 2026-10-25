
"""Faithful file selection for the Chapter 4 UNSW-NB15 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ATTACKER_IPS = tuple(f"175.45.176.{index}" for index in range(4))
VICTIM_IPS = tuple(f"149.171.126.{index}" for index in range(10, 20))
NORMAL_DESTINATION_IPS = tuple(f"149.171.126.{index}" for index in range(10))
NORMAL_SENDER_IPS = tuple(f"59.166.0.{index}" for index in range(10))


@dataclass(frozen=True)
class DatasetFile:
    """One directional per-IP CSV selected by the legacy experiment."""

    split: str
    role: str
    ip_address: str
    path: Path


@dataclass(frozen=True)
class LegacyDatasetManifest:
    """Training and test files used by the supplied Chapter 4 code."""

    training: tuple[DatasetFile, ...]
    testing_normal: tuple[DatasetFile, ...]
    testing_victim: tuple[DatasetFile, ...]

    @property
    def testing(self) -> tuple[DatasetFile, ...]:
        return self.testing_normal + self.testing_victim

    @property
    def all_files(self) -> tuple[DatasetFile, ...]:
        return self.training + self.testing

    def missing_files(self) -> tuple[Path, ...]:
        return tuple(item.path for item in self.all_files if not item.path.is_file())


def build_legacy_manifest(
    dataset_root: str | Path, *, require_files: bool = False
) -> LegacyDatasetManifest:
    """Build the original IP-based directional training/test selection."""

    root = Path(dataset_root)
    manifest = LegacyDatasetManifest(
        training=tuple(
            DatasetFile(
                split="train",
                role="normal_sender",
                ip_address=ip_address,
                path=root / ip_address / "flow_from_this.csv",
            )
            for ip_address in NORMAL_SENDER_IPS
        ),
        testing_normal=tuple(
            DatasetFile(
                split="test",
                role="normal_destination",
                ip_address=ip_address,
                path=root / ip_address / "flow_to_this.csv",
            )
            for ip_address in NORMAL_DESTINATION_IPS
        ),
        testing_victim=tuple(
            DatasetFile(
                split="test",
                role="victim",
                ip_address=ip_address,
                path=root / ip_address / "flow_to_this.csv",
            )
            for ip_address in VICTIM_IPS
        ),
    )

    if require_files:
        missing = manifest.missing_files()
        if missing:
            preview = "\n".join(str(path) for path in missing[:5])
            remainder = len(missing) - min(len(missing), 5)
            suffix = f"\n... and {remainder} more" if remainder else ""
            raise FileNotFoundError(
                f"Missing {len(missing)} legacy dataset files:\n{preview}{suffix}"
            )

    return manifest
