"""The record's daily export into the archive, on every backend.

Each closed arrival day lands once in its own day directory: one Parquet file
and the day's head under ``agent_record/``, and one ``agent_record_day`` line,
last, in that day's manifest. The day line is the only mark. The head is the
day's entries folded from the head of the day before
(:mod:`osprey.agent_record.chain`), the same on every backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from osprey.agent_record import archive, chain
from osprey.agent_record import contract as c
from osprey.services.archive.manifest import MANIFEST_NAME
from osprey.services.archive.run import ArchiveTree
from osprey.utils.workspace import ARCHIVE_DIR_RELPATH

from . import _kit as k

DAY = 24 * 3600
D0 = k.T0.date()
D1 = D0 + timedelta(days=1)
D2 = D0 + timedelta(days=2)
D3 = D0 + timedelta(days=3)

#: What a day with entries holds under ``agent_record/``.
BOTH = {archive.EXPORT_FILE, chain.HEAD_FILE}


def _root(tmp_path: Path, mode: int = 0o700) -> Path:
    root = tmp_path / "archive"
    root.mkdir()
    os.chmod(root, mode)
    return root


def _now(day: date) -> datetime:
    """Noon UTC of *day*: every day before it is closed."""
    return k.T0 + timedelta(days=(day - D0).days)


def _lines(root: Path, day) -> list[dict]:
    manifest = root / day.isoformat() / MANIFEST_NAME
    if not manifest.exists():
        return []
    return [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]


def _parquet_keys(path: Path) -> list[str]:
    con = duckdb.connect()
    try:
        return [
            r[0] for r in con.execute("SELECT key FROM read_parquet(?)", [str(path)]).fetchall()
        ]
    finally:
        con.close()


def _export_files(root: Path, day) -> list[Path]:
    sub = root / day.isoformat() / archive.EXPORT_SUBDIR
    return sorted(p for p in sub.iterdir()) if sub.is_dir() else []


def _names(root: Path, day) -> set[str]:
    return {p.name for p in _export_files(root, day)}


def _export(root: Path, day) -> Path:
    return root / day.isoformat() / archive.EXPORT_SUBDIR / archive.EXPORT_FILE


def _head_file(root: Path, day) -> Path:
    return root / day.isoformat() / archive.EXPORT_SUBDIR / chain.HEAD_FILE


def _head(root: Path, day) -> chain.DayHead:
    return chain.DayHead.from_json(_head_file(root, day).read_text())


def _feedback(key: str, arrived: float) -> c.Entry:
    return k.entry(k.ALICE, key, c.KIND_FEEDBACK, arrived, {"text": key}, arrived=arrived)


def test_each_closed_day_gets_one_parquet_its_head_and_one_mark_last(backend, settle, tmp_path):
    backend.append(
        [
            _feedback("sp:alice:a", 0),
            _feedback("sp:alice:b", 60),
            _feedback("sp:alice:c", DAY + 1),
            _feedback("sp:alice:today", 2 * DAY + 1),
        ]
    )
    settle()
    root = _root(tmp_path)

    assert archive.run_export(backend, _now(D2), root) == [D0, D1]

    for day, keys in ((D0, {"sp:alice:a", "sp:alice:b"}), (D1, {"sp:alice:c"})):
        assert _names(root, day) == BOTH
        assert set(_parquet_keys(_export(root, day))) == keys
        lines = _lines(root, day)
        assert [line["kind"] for line in lines] == ["file", "file", "agent_record_day"]
        file_line, head_line, mark = lines
        assert file_line["path"] == f"{day.isoformat()}/agent_record/{archive.EXPORT_FILE}"
        assert file_line["rows"] == len(keys)
        assert file_line["size"] == _export(root, day).stat().st_size
        assert len(file_line["sha256"]) == 64
        head_bytes = _head_file(root, day).read_bytes()
        assert head_line["path"] == f"{day.isoformat()}/agent_record/{chain.HEAD_FILE}"
        assert head_line["size"] == len(head_bytes)
        assert head_line["sha256"] == hashlib.sha256(head_bytes).hexdigest()
        assert _head(root, day).n == len(keys)
        assert mark == {"kind": "agent_record_day", "day": day.isoformat(), "rows": len(keys)}
    assert not (root / D2.isoformat()).exists()
    assert archive.exported_days(root) == {D0, D1}


def test_each_day_head_folds_its_entries_from_the_head_of_the_day_before(backend, settle, tmp_path):
    entries = {
        D0: [_feedback("sp:alice:b", 60), _feedback("sp:alice:a", 60), _feedback("sp:alice:c", 0)],
        D2: [_feedback("sp:alice:d", 2 * DAY + 1)],
    }
    backend.append([e for day in entries.values() for e in day])
    settle()
    root = _root(tmp_path)

    assert archive.run_export(backend, _now(D3), root) == [D0, D1, D2]

    # Computed from the entries alone, never from what the backend stored.
    prev = chain.ZERO_HEAD
    for day in (D0, D1, D2):
        folded = chain.fold(day, [c.entry_row(e, k.NOW) for e in entries.get(day, [])], prev)
        assert _head(root, day) == chain.DayHead(
            day=day.isoformat(),
            schema=archive.HEAD_SCHEMA,
            n=folded.n,
            prev=prev.hex(),
            head=folded.head.hex(),
        )
        prev = folded.head
    assert _head(root, D1).head == chain.genesis(D1, bytes.fromhex(_head(root, D0).head)).hex()


def test_an_exported_day_is_never_exported_again(backend, settle, tmp_path):
    backend.append([_feedback("sp:alice:a", 0)])
    settle()
    root = _root(tmp_path)
    assert archive.run_export(backend, _now(D1), root) == [D0]
    before = (root / D0.isoformat() / MANIFEST_NAME).read_bytes()

    assert archive.run_export(backend, _now(D1), root) == []
    assert archive.run_export(backend, _now(D1) + timedelta(hours=11), root) == []

    assert (root / D0.isoformat() / MANIFEST_NAME).read_bytes() == before
    assert _names(root, D0) == BOTH


def test_a_closed_day_without_entries_gets_its_head_and_no_export(backend, settle, tmp_path):
    backend.append([_feedback("sp:alice:a", 0), _feedback("sp:alice:c", 2 * DAY + 1)])
    settle()
    root = _root(tmp_path)

    assert archive.run_export(backend, _now(D3), root) == [D0, D1, D2]

    assert _names(root, D1) == {chain.HEAD_FILE}
    head_line, mark = _lines(root, D1)
    assert head_line["path"] == f"{D1.isoformat()}/agent_record/{chain.HEAD_FILE}"
    assert mark == {"kind": "agent_record_day", "day": D1.isoformat(), "rows": 0}
    head = _head(root, D1)
    assert (head.n, head.prev) == (0, _head(root, D0).head)
    assert head.head == chain.genesis(D1, bytes.fromhex(head.prev)).hex()
    assert _head(root, D2).prev == head.head
    assert list((root / ".incoming").iterdir()) == []
    assert archive.exported_days(root) == {D0, D1, D2}


def test_an_empty_record_exports_nothing(backend, tmp_path):
    root = _root(tmp_path)
    assert archive.run_export(backend, _now(D3), root) == []
    assert archive.exported_days(root) == set()


def test_catch_up_exports_every_missed_day_oldest_first(backend, settle, tmp_path):
    backend.append([_feedback(f"sp:alice:d{n}", n * DAY + 5) for n in range(4)])
    settle()
    root = _root(tmp_path)
    assert archive.run_export(backend, _now(D1), root) == [D0]

    # Three runs were missed; the next one exports the days they would have.
    assert archive.run_export(backend, _now(D3) + timedelta(days=1), root) == [D1, D2, D3]

    for n, day in enumerate((D0, D1, D2, D3)):
        assert _parquet_keys(_export(root, day)) == [f"sp:alice:d{n}"]
    # The later run's days link on from the head the first run left.
    for before, day in ((D0, D1), (D1, D2), (D2, D3)):
        assert _head(root, day).prev == _head(root, before).head
    assert archive.exported_days(root) == {D0, D1, D2, D3}


def test_two_concurrent_runs_export_a_day_once(backend, settle, tmp_path):
    backend.append([_feedback("sp:alice:a", 0), _feedback("sp:alice:c", DAY + 1)])
    settle()
    root = _root(tmp_path)
    start = threading.Barrier(2)
    results: list[list] = []
    errors: list[BaseException] = []

    def run() -> None:
        start.wait()
        try:
            results.append(archive.run_export(backend, _now(D2), root))
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not errors
    assert sorted(results, key=len) == [[], [D0, D1]]
    for day in (D0, D1):
        assert [line["kind"] for line in _lines(root, day)] == ["file", "file", "agent_record_day"]
        assert _names(root, day) == BOTH


def test_files_take_the_root_mode_without_setgid_or_exec(backend, settle, tmp_path):
    backend.append([_feedback("sp:alice:a", 0)])
    settle()
    root = _root(tmp_path, 0o2770)
    if stat.S_IMODE(root.stat().st_mode) != 0o2770:
        pytest.skip("this filesystem does not keep setgid on a directory")

    archive.run_export(backend, _now(D1), root)

    day_dir = root / D0.isoformat()
    for directory in (day_dir, day_dir / archive.EXPORT_SUBDIR):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o2770
    for path in (day_dir / MANIFEST_NAME, *_export_files(root, D0)):
        assert stat.S_IMODE(path.stat().st_mode) == 0o660


def test_a_crash_before_the_day_line_is_exported_again_in_place(
    backend, settle, tmp_path, monkeypatch
):
    backend.append([_feedback("sp:alice:a", 0)])
    settle()
    root = _root(tmp_path)
    real_append = ArchiveTree.append

    def crash_at_the_mark(self, record, **kwargs):
        if record["kind"] == "agent_record_day":
            raise OSError("disk went away")
        return real_append(self, record, **kwargs)

    monkeypatch.setattr(ArchiveTree, "append", crash_at_the_mark)
    with pytest.raises(OSError, match="disk went away"):
        archive.run_export(backend, _now(D1), root)
    assert archive.exported_days(root) == set()
    assert _names(root, D0) == BOTH

    backend.append([_feedback("sp:alice:late", 60)])
    settle()
    monkeypatch.setattr(ArchiveTree, "append", real_append)
    assert archive.run_export(backend, _now(D1), root) == [D0]

    assert _names(root, D0) == BOTH
    assert set(_parquet_keys(_export(root, D0))) == {"sp:alice:a", "sp:alice:late"}
    assert _head(root, D0).n == 2
    kinds = [line["kind"] for line in _lines(root, D0)]
    assert kinds == ["file", "file", "file", "file", "agent_record_day"]
    assert _lines(root, D0)[-1]["rows"] == 2
    assert list((root / ".incoming").iterdir()) == []


def test_a_day_whose_export_does_not_fold_to_its_count_stays_unmarked(
    backend, settle, tmp_path, monkeypatch
):
    backend.append([_feedback("sp:alice:a", 0), _feedback("sp:alice:b", 60)])
    settle()
    root = _root(tmp_path)
    real_read = archive.read_export
    monkeypatch.setattr(archive, "read_export", lambda path: real_read(path)[:-1])

    with pytest.raises(c.RecordBackendError, match="not the 2 counted"):
        archive.run_export(backend, _now(D1), root)

    assert archive.exported_days(root) == set()
    assert _export_files(root, D0) == []
    assert list((root / ".incoming").iterdir()) == []


def test_a_day_before_whose_head_is_not_a_head_fails_the_next_day(backend, settle, tmp_path):
    backend.append([_feedback("sp:alice:a", 0), _feedback("sp:alice:c", DAY + 1)])
    settle()
    root = _root(tmp_path)
    assert archive.run_export(backend, _now(D1), root) == [D0]
    _head_file(root, D0).write_text("{}\n")

    with pytest.raises(ValueError, match="is not a day head"):
        archive.run_export(backend, _now(D2), root)

    assert archive.exported_days(root) == {D0}


def test_a_purged_day_is_no_longer_exported(backend, settle, tmp_path):
    backend.append([_feedback("sp:alice:a", 0)])
    settle()
    root = _root(tmp_path)
    archive.run_export(backend, _now(D1), root)
    manifest = root / D0.isoformat() / MANIFEST_NAME
    with manifest.open("a") as handle:
        handle.write(json.dumps({"kind": "agent_record_purged", "day": D0.isoformat()}) + "\n")

    assert archive.exported_days(root) == set()


def test_the_export_writes_into_the_host_backed_archive(backend, settle, tmp_path):
    project = tmp_path / "project"
    root = project / ARCHIVE_DIR_RELPATH
    root.mkdir(parents=True)
    os.chmod(root, 0o750)
    backend.append([_feedback("sp:alice:a", 0)])
    settle()

    assert archive.run_export(backend, _now(D1), root) == [D0]

    exported = project / ARCHIVE_DIR_RELPATH / D0.isoformat() / "agent_record"
    assert {p.name for p in exported.iterdir()} == BOTH
    for name in BOTH:
        assert stat.S_IMODE((exported / name).stat().st_mode) == 0o640
    assert archive.exported_days(root) == {D0}


def test_an_archive_root_that_is_missing_fails_the_run(backend, settle, tmp_path):
    from osprey.services.archive.run import ArchiveDestinationError

    backend.append([_feedback("sp:alice:a", 0)])
    settle()
    with pytest.raises(ArchiveDestinationError):
        archive.run_export(backend, _now(D1), tmp_path / "absent")
