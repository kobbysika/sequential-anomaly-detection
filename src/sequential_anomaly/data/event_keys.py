
"""Faithful event-key construction for the Chapter 4 UNSW-NB15 experiment."""

from collections.abc import Sequence

from sequential_anomaly.data.schema import COLUMN_INDEX, validate_row_width


KEYSETS = (
    (
        "proto",
        "state",
        "service",
        "is_ftp_login",
        "ct_ftp_cmd",
        "ct_flw_http_mthd",
        "trans_depth",
        "is_sm_ips_ports",
        "packet_bucket",
    ),
    (
        "proto",
        "state",
        "service",
        "is_ftp_login",
        "ct_ftp_cmd",
        "ct_flw_http_mthd",
        "trans_depth",
        "is_sm_ips_ports",
    ),
    (
        "proto",
        "state",
        "service",
        "is_ftp_login",
        "ct_ftp_cmd",
        "ct_flw_http_mthd",
        "trans_depth",
    ),
    ("proto", "state", "service", "is_ftp_login", "ct_ftp_cmd"),
    ("proto", "state", "service", "is_ftp_login"),
)


def _integer_text(value: str) -> str:
    """Match the legacy conversion of blank numeric fields to zero."""

    value = value.strip()
    return str(int(value)) if value else "0"


def build_event_key(
    row: Sequence[str], *, keyset: int = 0, packet_divisor: int = 100
) -> str:
    """Convert one raw UNSW-NB15 row to a Chapter 4 categorical event key."""

    validate_row_width(list(row))
    if not 0 <= keyset < len(KEYSETS):
        raise ValueError(f"keyset must be between 0 and {len(KEYSETS) - 1}")
    if packet_divisor <= 0:
        raise ValueError("packet_divisor must be greater than zero")

    values = {name: row[index].strip() for name, index in COLUMN_INDEX.items()}
    values["is_ftp_login"] = str(int(_integer_text(values["is_ftp_login"])) & 1)
    values["ct_ftp_cmd"] = _integer_text(values["ct_ftp_cmd"])
    values["ct_flw_http_mthd"] = _integer_text(values["ct_flw_http_mthd"])
    values["trans_depth"] = _integer_text(values["trans_depth"])
    values["packet_bucket"] = str(
        (int(values["spkts"]) + int(values["dpkts"])) // packet_divisor
    )

    return ",".join(values[field] for field in KEYSETS[keyset])


def attack_category(row: Sequence[str]) -> str | None:
    """Return None for normal traffic or the named attack category."""

    validate_row_width(list(row))
    if row[COLUMN_INDEX["label"]].strip() == "0":
        return None
    return row[COLUMN_INDEX["attack_cat"]].strip()
