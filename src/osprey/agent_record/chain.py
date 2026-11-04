"""The day chain — one hash over each exported arrival day, linked day to day.

Each closed arrival day the archive exports (:mod:`osprey.agent_record.archive`)
is folded into one 32-byte *day head*:

- every stored row is a *leaf*: SHA-256 over the tag ``osprey-record/leaf/v1``
  and its :data:`~osprey.agent_record.contract.ENTRY_COLUMNS`, in that order,
  each prefixed with its length (an unsigned LEB128 varint);
- the fold starts at the day's *genesis*, SHA-256 over the tag
  ``osprey-record/day/v1``, the day and the previous day's head (32 zero bytes
  for the first day), and links every leaf in export order
  (``inserted_at, key, chunk_index``): ``head_i = SHA-256(link tag || head_{i-1}
  || leaf_i)``;
- the day head is the last link, and the genesis alone for a day without
  entries.

**Canonical form.** A leaf is computed from the export alone and is the same on
every backend: ``chunk_index`` and ``chunk_total`` are their decimal digits,
``ts`` and ``inserted_at`` the UTC time as ``YYYY-MM-DDTHH:MM:SS.ffffffZ``,
``body`` the stored :func:`~osprey.agent_record.contract.serialize_body` text
byte for byte, and every other column its UTF-8 bytes. No stored column is ever
NULL; one that is is refused rather than hashed.

**Order.** :func:`fold` sorts the rows into export order before it links them,
so the head commits to the day's set of rows, not to how a copy happens to
store them; ``(key, chunk_index)`` is unique, so the order is total.

**Who signs.** Nobody here. The host writes each day's :class:`DayHead`
unsigned beside the export (:data:`HEAD_FILE`); a collector off the host seals
the first head it receives for a day, signing :func:`seal_message`, and a
verifier recomputes every day from a copy of the archive and checks it
against the seals.

This module is standard library only and does no I/O.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from osprey.agent_record.contract import ENTRY_COLUMNS

__all__ = [
    "DAY_TAG",
    "HEAD_FILE",
    "HEAD_VERSION",
    "LEAF_TAG",
    "LINK_TAG",
    "SEAL_TAG",
    "ZERO_HEAD",
    "DayHead",
    "Fold",
    "SealedHead",
    "export_order_key",
    "fold",
    "genesis",
    "leaf",
    "link",
    "seal_message",
    "wire_time",
]

LEAF_TAG = b"osprey-record/leaf/v1\x00"
DAY_TAG = b"osprey-record/day/v1\x00"
LINK_TAG = b"osprey-record/link/v1\x00"
#: The tag of the bytes a collector signs (:func:`seal_message`).
SEAL_TAG = b"osprey-record/seal/v1\x00"

#: The previous head of a day that has no previous day.
ZERO_HEAD = bytes(32)

#: The version of :class:`DayHead` and of the tags above.
HEAD_VERSION = 1

#: A day's head, beside its export under the day's ``agent_record/``.
HEAD_FILE = "day_head.json"

_INT_COLUMNS = frozenset({"chunk_index", "chunk_total"})
_TIME_COLUMNS = frozenset({"ts", "inserted_at"})
_WIRE_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")
_HEX32 = re.compile(r"[0-9a-f]{64}")
_HEX64 = re.compile(r"[0-9a-f]{128}")


def wire_time(value: datetime | str) -> str:
    """*value* as the UTC time ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` a leaf commits to.

    An aware ``datetime`` is converted to UTC; a string must already be that
    form.

    Raises:
        ValueError: If *value* is a naive ``datetime`` or any other string.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(f"time {value!r} has no timezone")
        u = value.astimezone(UTC)
        return (
            f"{u.year:04d}-{u.month:02d}-{u.day:02d}T{u.hour:02d}:{u.minute:02d}:"
            f"{u.second:02d}.{u.microsecond:06d}Z"
        )
    if isinstance(value, str) and _WIRE_TIME.fullmatch(value):
        return value
    raise ValueError(f"time {value!r} is neither an aware datetime nor YYYY-MM-DDTHH:MM:SS.ffffffZ")


def _uvarint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field_bytes(column: str, value: Any) -> bytes:
    if value is None:
        raise ValueError(f"column {column} is NULL; no stored column ever is")
    if column in _INT_COLUMNS:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"column {column} holds {value!r}, not a count")
        return str(value).encode("ascii")
    if column in _TIME_COLUMNS:
        return wire_time(value).encode("ascii")
    if not isinstance(value, str):
        raise ValueError(f"column {column} holds {type(value).__name__}, not text")
    return value.encode("utf-8")


def _length_prefixed(tag: bytes, fields: Iterable[bytes]) -> bytes:
    out = bytearray(tag)
    for field in fields:
        out += _uvarint(len(field))
        out += field
    return bytes(out)


def leaf(row: Mapping[str, Any]) -> bytes:
    """The 32-byte leaf of one stored row of :data:`ENTRY_COLUMNS`.

    Raises:
        KeyError: If *row* lacks a column.
        ValueError: If a column's value has no canonical form.
    """
    return hashlib.sha256(
        _length_prefixed(LEAF_TAG, (_field_bytes(c, row[c]) for c in ENTRY_COLUMNS))
    ).digest()


def _day_label(day: date | str) -> str:
    label = day if isinstance(day, str) else day.isoformat()
    if date.fromisoformat(label).isoformat() != label:
        raise ValueError(f"day {day!r} is not YYYY-MM-DD")
    return label


def genesis(day: date | str, prev: bytes = ZERO_HEAD) -> bytes:
    """The fold's start: the day bound to the previous day's head."""
    if len(prev) != 32:
        raise ValueError(f"a previous head is 32 bytes, not {len(prev)}")
    return hashlib.sha256(DAY_TAG + _day_label(day).encode("ascii") + prev).digest()


def link(head: bytes, leaf_hash: bytes) -> bytes:
    """The next head: *head* linked to one more leaf."""
    return hashlib.sha256(LINK_TAG + head + leaf_hash).digest()


def export_order_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    """*row*'s place in the export order ``inserted_at, key, chunk_index``."""
    return (wire_time(row["inserted_at"]), row["key"], row["chunk_index"])


@dataclass(frozen=True)
class Fold:
    """One day folded: its head, its row count and its leaves in export order."""

    head: bytes
    n: int
    leaves: tuple[bytes, ...]


def fold(day: date | str, rows: Iterable[Mapping[str, Any]], prev: bytes = ZERO_HEAD) -> Fold:
    """Fold the day's *rows*, sorted into export order, from the day's genesis.

    Raises:
        KeyError: If a row lacks a column.
        ValueError: If a value has no canonical form, or *prev* is not 32 bytes.
    """
    head = genesis(day, prev)
    leaves = tuple(leaf(row) for row in sorted(rows, key=export_order_key))
    for leaf_hash in leaves:
        head = link(head, leaf_hash)
    return Fold(head, len(leaves), leaves)


def _hex(value: Any, pattern: re.Pattern[str], what: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{what} {value!r} is not lowercase hex of the right length")
    return value


@dataclass(frozen=True)
class DayHead:
    """The unsigned head of one arrival day, as the host writes it.

    Attributes:
        day: The arrival day, ``YYYY-MM-DD``.
        schema: The fingerprint of the record schema the leaves were computed
            under (hex SHA-256).
        n: The day's row count.
        prev: The previous day's head (hex), zeros for the first day.
        head: The day head (hex).
        v: The version of this record and of the tags.
    """

    day: str
    schema: str
    n: int
    prev: str
    head: str
    v: int = HEAD_VERSION

    def __post_init__(self) -> None:
        if self.v != HEAD_VERSION:
            raise ValueError(f"head version {self.v!r} is not {HEAD_VERSION}")
        _day_label(self.day)
        _hex(self.schema, _HEX32, "schema")
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 0:
            raise ValueError(f"row count {self.n!r} is not a count")
        _hex(self.prev, _HEX32, "prev")
        _hex(self.head, _HEX32, "head")

    def fields(self) -> dict[str, Any]:
        """The head's fields in record order."""
        return {
            "v": self.v,
            "day": self.day,
            "schema": self.schema,
            "n": self.n,
            "prev": self.prev,
            "head": self.head,
        }

    def to_json(self) -> str:
        """One JSON line, newline included."""
        return json.dumps(self.fields(), separators=(",", ":")) + "\n"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DayHead:
        """The head *data* holds; any other key is ignored.

        Raises:
            ValueError: If a field is missing or malformed.
        """
        try:
            return cls(
                v=data["v"],
                day=data["day"],
                schema=data["schema"],
                n=data["n"],
                prev=data["prev"],
                head=data["head"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"not a day head: {exc!r}") from None

    @classmethod
    def from_json(cls, text: str) -> DayHead:
        """The head a JSON object holds.

        Raises:
            ValueError: If *text* is not one.
        """
        data = json.loads(text)
        if not isinstance(data, Mapping):
            raise ValueError("a day head is a JSON object")
        return cls.from_mapping(data)


def seal_message(head: DayHead, sealed_at: str) -> bytes:
    """The bytes a collector signs for *head*, sealed at *sealed_at*.

    The head's fields in record order and then the sealing time, each as
    ASCII text, length-prefixed under :data:`SEAL_TAG` as a leaf's columns are.
    """
    fields = [str(value) for value in head.fields().values()]
    fields.append(wire_time(sealed_at))
    return _length_prefixed(SEAL_TAG, (field.encode("ascii") for field in fields))


@dataclass(frozen=True)
class SealedHead:
    """A day head as a collector sealed it.

    Attributes:
        head: The head the host sent.
        sealed_at: When the collector sealed it (UTC, ``…ffffffZ``).
        sig: The Ed25519 signature over :func:`seal_message` (hex).
        pub: The collector's Ed25519 public key (hex). A verifier pins the key
            it expects and never takes this field's word for it.
    """

    head: DayHead
    sealed_at: str
    sig: str
    pub: str

    def __post_init__(self) -> None:
        wire_time(self.sealed_at)
        _hex(self.sig, _HEX64, "sig")
        _hex(self.pub, _HEX32, "pub")

    def to_json(self) -> str:
        """One JSON line, newline included."""
        record = {
            **self.head.fields(),
            "sealed_at": self.sealed_at,
            "sig": self.sig,
            "pub": self.pub,
        }
        return json.dumps(record, separators=(",", ":")) + "\n"

    @classmethod
    def from_json(cls, text: str) -> SealedHead:
        """The sealed head a JSON object holds.

        Raises:
            ValueError: If *text* is not one.
        """
        data = json.loads(text)
        if not isinstance(data, Mapping):
            raise ValueError("a sealed head is a JSON object")
        try:
            return cls(
                head=DayHead.from_mapping(data),
                sealed_at=data["sealed_at"],
                sig=data["sig"],
                pub=data["pub"],
            )
        except KeyError as exc:
            raise ValueError(f"not a sealed head: {exc!r}") from None
