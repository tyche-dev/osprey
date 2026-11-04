"""The daily export of the record to the archive.

:func:`run_export` is the scheduler's first daily job
(:mod:`osprey.agent_record.scheduler`), which looks it up on this module by
name at every run.

Each closed arrival day (a UTC day before the run's day) is exported once,
into that day's own directory of the archive root:
``<root>/<day>/agent_record/entries.parquet`` holds every entry that arrived
that day, and ``day_head.json`` beside it the day's head
(:class:`~osprey.agent_record.chain.DayHead`): the file's rows folded, in
export order, from the head of the day before (:mod:`osprey.agent_record.chain`).
The day's ``MANIFEST.jsonl`` gets a ``file`` line for each and then an
``agent_record_day`` line, last. A day with no entries gets its head (the
genesis alone, ``n: 0``) and the day line with ``rows: 0``, and no export
file. The day line is the only mark of an exported day (:func:`exported_days`);
an ``agent_record_purged`` line clears it.

The head is unsigned, and the host never signs it: a collector off the host
seals it. The head of the day before is read from the archive; where the
archive holds none (the first day, or a day exported before heads were
written), the day starts from 32 zero bytes.

The export writes through the archive service's
:class:`~osprey.services.archive.run.ArchiveTree`: it holds the one archive
lock on ``.incoming`` from before it reads the marks until its last line is
written, so two runs never export a day twice, and every file is staged in
``.incoming`` and moved into place only when whole. A run that stopped between
placing a day's file and writing its day line left that day unmarked; the next
run exports the day again and replaces that file rather than adding a second.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from osprey.agent_record import chain
from osprey.agent_record.backends import clickhouse_ddl
from osprey.agent_record.contract import ENTRY_COLUMNS, RecordBackend, RecordBackendError
from osprey.services.archive.manifest import AGENT_RECORD_DAY, load_state
from osprey.services.archive.run import ArchiveTree, file_record

__all__ = [
    "EXPORT_FILE",
    "EXPORT_SUBDIR",
    "HEAD_SCHEMA",
    "exported_days",
    "oldest_arrival_day",
    "read_export",
    "run_export",
    "unexported_days",
]

#: The directory of a day's export, under that day's directory.
EXPORT_SUBDIR = "agent_record"

#: A day's export file, under :data:`EXPORT_SUBDIR`.
EXPORT_FILE = "entries.parquet"

#: The schema a day head names: the record table's fingerprint, the ``COMMENT``
#: the ClickHouse stage stamps on the live table. The embedded backend exports
#: the same columns, so a head names the same schema on either backend, and a
#: migration that changes a column changes it.
HEAD_SCHEMA = clickhouse_ddl.fingerprint(clickhouse_ddl.ENTRIES)


def exported_days(archive_root: Path) -> set[date]:
    """The arrival days whose latest record line in the archive is ``agent_record_day``.

    Raises:
        OSError: If a manifest exists but cannot be read.
    """
    return {date.fromisoformat(day) for day in load_state(archive_root).agent_record_days}


def unexported_days(oldest: date | None, today: date, done: set[date]) -> list[date]:
    """The closed days from *oldest* through the day before *today* not in *done*, oldest first.

    Empty when *oldest* is ``None``, a record that holds no entry.
    """
    days: list[date] = []
    day = oldest
    while day is not None and day < today:
        if day not in done:
            days.append(day)
        day += timedelta(days=1)
    return days


def oldest_arrival_day(record: RecordBackend) -> date | None:
    """The oldest arrival day *record* holds; ``None`` when it holds no entry.

    Raises:
        RecordBackendError: If the record's health is not ok, so it cannot say.
    """
    health = record.health()
    if not health.ok:
        raise RecordBackendError(f"the record cannot say which days it holds: {health.detail}")
    return health.oldest_arrival_day


def read_export(path: Path) -> list[dict[str, Any]]:
    """The rows of one day's export file, :data:`ENTRY_COLUMNS` in that order.

    Read with DuckDB in UTC, so ``ts`` and ``inserted_at`` are aware times.

    Raises:
        ValueError: If *path* cannot be read as Parquet, or its columns are not
            :data:`ENTRY_COLUMNS` in that order.
    """
    import duckdb

    con = duckdb.connect()
    try:
        con.execute("SET TimeZone = 'UTC'")
        cursor = con.execute("SELECT * FROM read_parquet(?)", [str(path)])
        columns = tuple(d[0] for d in cursor.description)
        if columns != ENTRY_COLUMNS:
            raise ValueError(f"{path} holds the columns {columns}, not {ENTRY_COLUMNS}")
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    except duckdb.Error as exc:
        raise ValueError(f"{path} cannot be read as an export: {exc}") from exc
    finally:
        con.close()


def _previous_head(archive_root: Path, day: date) -> bytes:
    """The head the archive holds for the day before *day*; zeros when it holds none.

    Raises:
        ValueError: If that day's head file is not a head.
    """
    before = (day - timedelta(days=1)).isoformat()
    path = archive_root / before / EXPORT_SUBDIR / chain.HEAD_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return chain.ZERO_HEAD
    try:
        head = chain.DayHead.from_json(text)
    except ValueError as exc:
        raise ValueError(f"{path} is not a day head: {exc}") from exc
    if head.day != before:
        raise ValueError(f"{path} is the head of {head.day}, not of {before}")
    return bytes.fromhex(head.head)


def _day_head(day: date, export: Path | None, rows: int, prev: bytes) -> chain.DayHead:
    """The head of *day*: the rows of *export* (none without one) folded from *prev*."""
    folded = chain.fold(day, read_export(export) if export is not None else [], prev)
    if folded.n != rows:
        raise RecordBackendError(
            f"the export of arrival day {day} holds {folded.n} entries, not the {rows} counted"
        )
    return chain.DayHead(
        day=day.isoformat(),
        schema=HEAD_SCHEMA,
        n=folded.n,
        prev=prev.hex(),
        head=folded.head.hex(),
    )


def _place_head(tree: ArchiveTree, head: chain.DayHead) -> None:
    temp, size, sha = tree.stage(iter([head.to_json().encode("utf-8")]))
    try:
        path = tree.place(temp, f"{EXPORT_SUBDIR}/{chain.HEAD_FILE}", day=head.day, replace=True)
    finally:
        tree.discard(temp)
    tree.append(
        file_record(path, source=path, size=size, sha256=sha, source_mtime_ns=None, day=head.day),
        day=head.day,
    )


def _export_day(record: RecordBackend, tree: ArchiveTree, day: date) -> None:
    label = day.isoformat()
    prev = _previous_head(tree.dest, day)
    temp = tree.staging_path(".parquet")
    try:
        rows = record.export_parquet(day, temp)
        head = _day_head(day, temp if rows else None, rows, prev)
        if rows:
            size, sha = tree.adopt(temp)
            path = tree.place(temp, f"{EXPORT_SUBDIR}/{EXPORT_FILE}", day=label, replace=True)
            tree.append(
                file_record(
                    path,
                    source=path,
                    size=size,
                    sha256=sha,
                    source_mtime_ns=None,
                    day=label,
                    rows=rows,
                ),
                day=label,
            )
    finally:
        tree.discard(temp)
    _place_head(tree, head)
    tree.append({"kind": AGENT_RECORD_DAY, "day": label, "rows": rows}, day=label)


def run_export(record: RecordBackend, now: datetime, archive_root: Path) -> list[date]:
    """Export every closed arrival day of *record* the archive does not hold yet.

    The days run from the record's oldest arrival day through the day before
    *now*'s UTC day, oldest first, skipping every day in
    :func:`exported_days`. The marks are read, and every day written, under the
    archive lock.

    Args:
        record: The record backend the service runs.
        now: The scheduler's clock reading for this run (aware UTC).
        archive_root: The archive root the days are exported into.

    Returns:
        The days exported, oldest first.

    Raises:
        RecordBackendError: If the record cannot be read, or a day's export
            does not hold the entries the record counted; the failing day and
            every later one stay unmarked.
        ValueError: If a day's export cannot be read back to fold, or the head
            of the day before it is not a head; the day stays unmarked too.
        osprey.services.archive.run.ArchiveDestinationError: If the archive
            root cannot be written.
        OSError: If the archive cannot be read or written part-way.
    """
    tree = ArchiveTree(archive_root, now)
    try:
        done = exported_days(archive_root)
        days = unexported_days(oldest_arrival_day(record), now.astimezone(UTC).date(), done)
        exported: list[date] = []
        for day in days:
            _export_day(record, tree, day)
            exported.append(day)
        return exported
    finally:
        tree.close()
