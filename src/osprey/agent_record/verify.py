"""The offline verifier — a copy of the archive checked against the collector's seals.

It needs nothing from the deploy host: a copy of the archive root
(``var/archive``), the collector's seals (its ``sealed/`` directory) and the
collector's public key, which whoever verifies pins themselves; the ``pub``
field of a seal is never taken on its word. Each sealed day, oldest first, is
accepted only when all four hold:

1. **signature** — the seal names the pinned key, and its Ed25519 signature
   over :func:`~osprey.agent_record.chain.seal_message` verifies under it;
2. **schema** — the seal names the schema this verifier reads
   (:data:`~osprey.agent_record.archive.HEAD_SCHEMA` unless told otherwise), so
   old leaves are never read under other rules;
3. **day link** — the day before is sealed too and the seal's ``prev`` is its
   head; only the first sealed day starts from zeros. The host exports and
   heads every day, idle ones included, so a day missing from the seals is a
   day dropped, and it breaks the link of the day after;
4. **rows** — the day's export in the copy, read and sorted into export order,
   holds the sealed ``n`` rows and folds from ``prev`` to the sealed head. A
   day sealed with ``n: 0`` has no export file.

A reorder of a day's rows in storage is not a change: the fold sorts first.
With a reference copy of the archive (the files as shipped off the host, say),
a day whose rows do not fold is compared row by row with the reference's, and
the first row where the two differ is named, when the reference's rows
themselves fold to the seal.

Usage::

    python -m osprey.agent_record.verify ARCHIVE SEALED --pub HEX [--reference ARCHIVE]

prints one ``ACCEPT`` or ``REJECT`` line per sealed day, then an ``UNSEALED``
line for each exported day in the copy that no seal covers, and exits 1 if
any day is rejected, 2 if the seals cannot be read.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from osprey.agent_record import chain
from osprey.agent_record.archive import EXPORT_FILE, EXPORT_SUBDIR, HEAD_SCHEMA, read_export

__all__ = [
    "DayVerdict",
    "first_divergence",
    "load_seals",
    "main",
    "unsealed_days",
    "verify_archive",
]

Row = Mapping[str, Any]


@dataclass(frozen=True)
class DayVerdict:
    """What the verifier found for one sealed day.

    Attributes:
        day: The arrival day.
        reasons: Why the day is rejected; empty when it is accepted.
        n: The rows the copy holds for the day, when it could be read.
        head: The head those rows fold to (hex), when it could be read.
    """

    day: str
    reasons: tuple[str, ...]
    n: int | None = None
    head: str | None = None

    @property
    def accepted(self) -> bool:
        return not self.reasons

    def line(self) -> str:
        if self.accepted:
            return f"ACCEPT {self.day} n={self.n} head={self.head}"
        return f"REJECT {self.day}: " + "; ".join(self.reasons)


def load_seals(directory: Path) -> list[chain.SealedHead]:
    """The seals ``<day>.json`` in *directory*, oldest day first.

    Raises:
        ValueError: If a file is not a seal, or not named for its own day.
    """
    seals = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            seal = chain.SealedHead.from_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(f"{path} is not a seal: {exc}") from exc
        if path.stem != seal.head.day:
            raise ValueError(f"{path} holds the seal of {seal.head.day}")
        seals.append(seal)
    return seals


def _short(hex_text: str) -> str:
    return f"{hex_text[:16]}…"


def _signature_reason(seal: chain.SealedHead, pub: str) -> str | None:
    if seal.pub != pub:
        return f"sealed by key {_short(seal.pub)}, not the pinned collector key {_short(pub)}"
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub)).verify(
            bytes.fromhex(seal.sig), chain.seal_message(seal.head, seal.sealed_at)
        )
    except InvalidSignature:
        return "the signature does not verify under the pinned key"
    return None


def _link_reason(seal: chain.SealedHead, before: chain.SealedHead | None) -> str | None:
    day_before = (date.fromisoformat(seal.head.day) - timedelta(days=1)).isoformat()
    if before is None:
        if seal.head.prev != chain.ZERO_HEAD.hex():
            return (
                f"broken day link: prev {_short(seal.head.prev)} is the head of {day_before}, "
                "and no earlier day is sealed"
            )
        return None
    if before.head.day != day_before:
        return f"broken day link: {day_before} is not sealed (the sealed day before is {before.head.day})"
    if seal.head.prev != before.head.head:
        return (
            f"broken day link: prev {_short(seal.head.prev)} is not the sealed head "
            f"{_short(before.head.head)} of {day_before}"
        )
    return None


def _day_rows(archive_root: Path, day: str) -> list[dict[str, Any]] | None:
    """The rows of *day*'s export in *archive_root*; ``None`` when there is no export file."""
    path = archive_root / day / EXPORT_SUBDIR / EXPORT_FILE
    if not path.is_file():
        return None
    return read_export(path)


def _unit(row: Row | None) -> str:
    return "no row" if row is None else f"{row['key']}#{row['chunk_index']}"


def first_divergence(rows: Sequence[Row], reference: Sequence[Row]) -> str | None:
    """The first row, in export order, where *rows* differ from *reference*; ``None`` if nowhere.

    Named by its position and ``key#chunk_index``, as edited (the same row,
    other content), missing (a reference row the copy lacks) or extra (a row
    the reference lacks).
    """
    ours = sorted(rows, key=chain.export_order_key)
    theirs = sorted(reference, key=chain.export_order_key)
    our_units = {_unit(r) for r in ours}
    their_units = {_unit(r) for r in theirs}
    for i in range(max(len(ours), len(theirs))):
        mine = ours[i] if i < len(ours) else None
        ref = theirs[i] if i < len(theirs) else None
        if mine is not None and ref is not None and chain.leaf(mine) == chain.leaf(ref):
            continue
        if mine is not None and ref is not None and _unit(mine) == _unit(ref):
            return f"row {i} ({_unit(mine)}) is edited"
        if ref is not None and _unit(ref) not in our_units:
            return f"row {i} ({_unit(ref)}) is missing"
        if mine is not None and _unit(mine) not in their_units:
            return f"row {i} ({_unit(mine)}) is extra"
        return f"row {i} differs ({_unit(mine)} here, {_unit(ref)} in the reference)"
    return None


def _rows_reasons(
    seal: chain.SealedHead,
    rows: list[dict[str, Any]] | None,
    reference: list[dict[str, Any]] | None,
) -> tuple[list[str], chain.Fold]:
    prev = bytes.fromhex(seal.head.prev)
    if rows is None and seal.head.n:
        missing = f"no export file for the {seal.head.n} sealed rows"
        return [missing], chain.fold(seal.head.day, [], prev)
    rows = rows or []
    folded = chain.fold(seal.head.day, rows, prev)
    reasons = []
    if folded.n != seal.head.n:
        side = "short" if folded.n < seal.head.n else "long"
        reasons.append(f"{folded.n} rows, sealed {seal.head.n} ({side})")
    if folded.head.hex() != seal.head.head:
        reasons.append(
            f"rows fold to {_short(folded.head.hex())}, not the sealed head {_short(seal.head.head)}"
        )
        if reference is not None and chain.fold(seal.head.day, reference, prev).head.hex() == (
            seal.head.head
        ):
            where = first_divergence(rows, reference)
            if where is not None:
                reasons.append(f"first divergent {where}")
    return reasons, folded


def verify_archive(
    archive_root: Path,
    seals: Sequence[chain.SealedHead],
    *,
    pub: str,
    schema: str = HEAD_SCHEMA,
    reference: Path | None = None,
) -> list[DayVerdict]:
    """Check every seal in *seals*, oldest first, against the copy *archive_root*.

    Args:
        archive_root: A copy of the archive root.
        seals: The collector's seals.
        pub: The collector's public key (hex), pinned by whoever verifies.
        schema: The schema this verifier reads leaves under.
        reference: Another copy of the archive, to name a day's first
            divergent row.

    Returns:
        One verdict per seal, oldest day first.
    """
    verdicts = []
    before: chain.SealedHead | None = None
    for seal in sorted(seals, key=lambda s: s.head.day):
        reasons = [
            reason
            for reason in (_signature_reason(seal, pub), _link_reason(seal, before))
            if reason is not None
        ]
        if seal.head.schema != schema:
            reasons.append(f"sealed under schema {_short(seal.head.schema)}, not {_short(schema)}")
        before = seal
        try:
            rows = _day_rows(archive_root, seal.head.day)
        except ValueError as exc:
            verdicts.append(DayVerdict(seal.head.day, (*reasons, f"export unreadable: {exc}")))
            continue
        ref = None
        if reference is not None:
            try:
                ref = _day_rows(reference, seal.head.day)
            except ValueError:
                ref = None
        row_reasons, folded = _rows_reasons(seal, rows, ref)
        verdicts.append(
            DayVerdict(seal.head.day, (*reasons, *row_reasons), folded.n, folded.head.hex())
        )
    return verdicts


def unsealed_days(archive_root: Path, seals: Sequence[chain.SealedHead]) -> list[str]:
    """The days the copy holds an export or a head for that no seal covers, oldest first."""
    sealed = {seal.head.day for seal in seals}
    days = []
    for path in sorted(Path(archive_root).glob(f"*/{EXPORT_SUBDIR}")):
        day = path.parent.name
        if day not in sealed and any(
            (path / name).is_file() for name in (EXPORT_FILE, chain.HEAD_FILE)
        ):
            days.append(day)
    return days


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m osprey.agent_record.verify",
        description="Check a copy of the archive against the collector's seals.",
    )
    parser.add_argument("archive", type=Path, help="a copy of the archive root")
    parser.add_argument("sealed", type=Path, help="the collector's sealed/ directory")
    parser.add_argument("--pub", required=True, help="the collector's public key, hex")
    parser.add_argument("--schema", default=HEAD_SCHEMA, help="the schema leaves are read under")
    parser.add_argument("--reference", type=Path, help="another copy, to name a divergent row")
    args = parser.parse_args(argv)

    if not args.sealed.is_dir():
        parser.error(f"{args.sealed} is not a directory")
    try:
        seals = load_seals(args.sealed)
    except ValueError as exc:
        parser.error(str(exc))
    verdicts = verify_archive(
        args.archive, seals, pub=args.pub, schema=args.schema, reference=args.reference
    )
    for verdict in verdicts:
        sys.stdout.write(verdict.line() + "\n")
    for day in unsealed_days(args.archive, seals):
        sys.stdout.write(f"UNSEALED {day}\n")
    return 0 if all(v.accepted for v in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
