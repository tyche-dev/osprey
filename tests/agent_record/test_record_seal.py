"""Sealing day heads off the host, and checking a copy of the archive against the seals.

An embedded record exports three closed days the way the scheduler does (the
middle one idle); a collector in its own directory seals each day's head; the
verifier checks a copy of the archive against those seals. The cases are the
design's: a faithful copy, idle day included (A1, A2); a row edited, deleted or
back-dated (R1, R2, R3); a day dropped from the copy or from the seals (R4); a
second head for a sealed day, or a seal under another key (R5); and a reorder
in storage, which is not a change (R6).
"""

from __future__ import annotations

import shutil
import stat
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path

import duckdb
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from osprey.agent_record import archive, chain, collector, verify
from osprey.agent_record import contract as c
from osprey.agent_record.backends.embedded import EmbeddedBackend
from tests.agent_record.contract_suite import _kit as k

DAY = 24 * 3600
D0 = k.T0.date()
D1 = D0 + timedelta(days=1)
D2 = D0 + timedelta(days=2)
D3 = D0 + timedelta(days=3)

#: Three entries on D0, none on D1, two on D2.
ENTRIES = [
    ("sp:alice:a", 0),
    ("sp:alice:b", 60),
    ("sp:alice:c", 120),
    ("sp:alice:d", 2 * DAY + 5),
    ("sp:alice:e", 2 * DAY + 65),
]


@dataclass(frozen=True)
class Sealed:
    """An archive as the host wrote it, a copy of it to check, and the collector's seals."""

    archive: Path
    copy: Path
    collector: collector.Collector
    seals: list[chain.SealedHead]


def _feedback(key: str, arrived: float) -> c.Entry:
    return k.entry(k.ALICE, key, c.KIND_FEEDBACK, arrived, {"text": key}, arrived=arrived)


def _export(root: Path, day) -> Path:
    return root / day.isoformat() / archive.EXPORT_SUBDIR / archive.EXPORT_FILE


def _head_file(root: Path, day) -> Path:
    return root / day.isoformat() / archive.EXPORT_SUBDIR / chain.HEAD_FILE


@pytest.fixture
def sealed(tmp_path) -> Sealed:
    root = tmp_path / "archive"
    root.mkdir(mode=0o700)
    with EmbeddedBackend(tmp_path / "record.duckdb") as backend:
        backend.append([_feedback(key, arrived) for key, arrived in ENTRIES])
        assert archive.run_export(backend, k.T0 + timedelta(days=3), root) == [D0, D1, D2]
    sealer = collector.Collector(tmp_path / "collector")
    for day in (D0, D1, D2):
        sealer.seal(chain.DayHead.from_json(_head_file(root, day).read_text()))
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    return Sealed(root, copy, sealer, verify.load_seals(sealer.root / collector.SEALED_DIR))


def _check(sealed: Sealed, **kwargs) -> list[str]:
    """The verifier's line for each sealed day of the copy."""
    seals = kwargs.pop("seals", sealed.seals)
    pub = kwargs.pop("pub", sealed.collector.public_key)
    return [v.line() for v in verify.verify_archive(sealed.copy, seals, pub=pub, **kwargs)]


def _rows(root: Path, day) -> list[dict]:
    return archive.read_export(_export(root, day))


def _write_rows(root: Path, day, rows: list[dict]) -> None:
    """Rewrite *day*'s export in *root* to hold *rows*, in that order, under the file's schema."""
    path = _export(root, day)
    temp = path.with_name(f"{path.name}.new")
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone = 'UTC'")
        con.execute("CREATE TABLE t AS SELECT * FROM read_parquet(?) LIMIT 0", [str(path)])
        marks = ", ".join("?" for _ in c.ENTRY_COLUMNS)
        con.executemany(
            f"INSERT INTO t VALUES ({marks})", [[r[col] for col in c.ENTRY_COLUMNS] for r in rows]
        )
        con.execute(f"COPY t TO '{temp}' (FORMAT PARQUET)")
    finally:
        con.close()
    temp.replace(path)


def _a_head(day: str = "2026-11-13") -> chain.DayHead:
    return chain.DayHead(
        day=day,
        schema=archive.HEAD_SCHEMA,
        n=0,
        prev=chain.ZERO_HEAD.hex(),
        head=chain.genesis(day).hex(),
    )


# --- the collector ----------------------------------------------------------------------------


def test_the_collector_makes_its_test_key_once_and_keeps_it_to_itself(tmp_path):
    root = tmp_path / "collector"
    first = collector.Collector(root)

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / collector.KEY_FILE).stat().st_mode) == 0o600
    assert collector.Collector(root).public_key == first.public_key
    assert len(bytes.fromhex(first.public_key)) == 32


def test_each_day_is_sealed_under_the_collector_key(sealed):
    key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(sealed.collector.public_key))
    assert [s.head.day for s in sealed.seals] == [D0.isoformat(), D1.isoformat(), D2.isoformat()]
    for seal in sealed.seals:
        assert seal.pub == sealed.collector.public_key
        key.verify(bytes.fromhex(seal.sig), chain.seal_message(seal.head, seal.sealed_at))
        path = sealed.collector.root / collector.SEALED_DIR / f"{seal.head.day}.json"
        assert stat.S_IMODE(path.stat().st_mode) == 0o644
        assert sealed.collector.sealed(seal.head.day) == seal


def test_r5_a_second_different_head_for_a_sealed_day_is_refused(sealed):
    kept = sealed.seals[0]
    path = sealed.collector.root / collector.SEALED_DIR / f"{D0}.json"
    before = path.read_bytes()
    doctored = replace(kept.head, head=chain.link(bytes.fromhex(kept.head.head), bytes(32)).hex())

    with pytest.raises(collector.SealRefused, match=f"{D0} is sealed with head {kept.head.head}"):
        sealed.collector.seal(doctored)
    # A collector restarted on the same directory refuses it too.
    with pytest.raises(collector.SealRefused):
        collector.Collector(sealed.collector.root).seal(doctored)

    assert path.read_bytes() == before


def test_the_same_head_again_gets_the_seal_already_kept(sealed):
    assert sealed.collector.seal(sealed.seals[0].head) == sealed.seals[0]


def test_of_two_writers_of_one_seal_file_the_first_wins(tmp_path):
    path = tmp_path / "seal.json"
    assert collector._write_new(path, b"first\n", 0o644)
    assert not collector._write_new(path, b"second\n", 0o644)
    assert path.read_bytes() == b"first\n"
    assert [p.name for p in tmp_path.iterdir()] == ["seal.json"]


def test_a_seal_is_synced_into_its_directory_before_it_is_returned(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(
        collector,
        "_fsync_dir",
        lambda d: synced.append((d, sorted(p.name for p in d.glob("*.json")))),
    )
    keeper = collector.Collector(tmp_path / "collector")
    head = _a_head()
    keeper.seal(head)
    sealed_dir = tmp_path / "collector" / collector.SEALED_DIR
    assert synced[-1] == (sealed_dir, [f"{head.day}.json"])


def test_a_collector_that_loses_the_race_for_a_day_keeps_the_winner(tmp_path, monkeypatch):
    root = tmp_path / "collector"
    first = collector.Collector(root)
    late = collector.Collector(root, clock=lambda: k.T0 + timedelta(days=30))
    real = late.sealed
    looks: list[str] = []

    def found_nothing_at_first(day: str) -> chain.SealedHead | None:
        # The late one looked before the first one wrote.
        looks.append(day)
        return None if len(looks) % 2 else real(day)

    monkeypatch.setattr(late, "sealed", found_nothing_at_first)
    won = first.seal(_a_head())

    with pytest.raises(collector.SealRefused):
        late.seal(replace(_a_head(), n=1))
    assert late.seal(_a_head()) == won
    assert first.sealed(won.head.day) == won


# --- the verifier -----------------------------------------------------------------------------


def test_a1_a2_a_faithful_copy_is_accepted_idle_day_included(sealed):
    assert _check(sealed) == [
        f"ACCEPT {D0} n=3 head={sealed.seals[0].head.head}",
        f"ACCEPT {D1} n=0 head={sealed.seals[1].head.head}",
        f"ACCEPT {D2} n=2 head={sealed.seals[2].head.head}",
    ]


def test_r1_an_edited_row_is_rejected_and_named(sealed):
    rows = _rows(sealed.copy, D0)
    rows[1] = {**rows[1], "body": rows[1]["body"] + " "}
    _write_rows(sealed.copy, D0, rows)

    lines = _check(sealed)
    assert lines[0].startswith(f"REJECT {D0}: rows fold to ")
    assert [line.split()[0] for line in lines[1:]] == ["ACCEPT", "ACCEPT"]
    named = _check(sealed, reference=sealed.archive)[0]
    assert named.endswith("; first divergent row 1 (sp:alice:b#0) is edited")


def test_r2_a_deleted_row_is_rejected_and_named(sealed):
    rows = _rows(sealed.copy, D0)
    _write_rows(sealed.copy, D0, [rows[0], rows[2]])

    line = _check(sealed, reference=sealed.archive)[0]
    assert line.startswith(f"REJECT {D0}: 2 rows, sealed 3 (short); rows fold to ")
    assert line.endswith("; first divergent row 1 (sp:alice:b#0) is missing")


def test_r3_a_back_dated_row_is_rejected_and_named(sealed):
    rows = _rows(sealed.copy, D0)
    extra = {
        **rows[0],
        "key": "sp:alice:z",
        "inserted_at": rows[0]["inserted_at"] + timedelta(seconds=30),
    }
    _write_rows(sealed.copy, D0, [*rows, extra])

    line = _check(sealed, reference=sealed.archive)[0]
    assert line.startswith(f"REJECT {D0}: 4 rows, sealed 3 (long); rows fold to ")
    assert line.endswith("; first divergent row 1 (sp:alice:z#0) is extra")


def test_r4_a_day_dropped_from_the_copy_is_rejected(sealed):
    shutil.rmtree(sealed.copy / D2.isoformat())

    assert _check(sealed)[2] == f"REJECT {D2}: no export file for the 2 sealed rows"


def test_r4_a_day_missing_from_the_seals_breaks_the_link_of_the_day_after(sealed):
    lines = _check(sealed, seals=[sealed.seals[0], sealed.seals[2]])

    assert lines[1] == (
        f"REJECT {D2}: broken day link: {D1} is not sealed (the sealed day before is {D0})"
    )


def test_r4_a_first_seal_that_names_a_day_before_is_rejected(sealed):
    lines = _check(sealed, seals=sealed.seals[1:])

    assert lines[0].startswith(f"REJECT {D1}: broken day link: prev ")
    assert lines[0].endswith(f"is the head of {D0}, and no earlier day is sealed")
    assert lines[1].startswith(f"ACCEPT {D2}")


def test_r5_seals_under_another_key_are_rejected(sealed, tmp_path):
    other = collector.Collector(tmp_path / "other")
    resealed = [other.seal(seal.head) for seal in sealed.seals]

    for lines in (_check(sealed, pub=other.public_key), _check(sealed, seals=resealed)):
        assert all(" not the pinned collector key " in line for line in lines), lines


def test_a_seal_whose_fields_were_changed_is_rejected(sealed):
    seals = list(sealed.seals)
    seals[2] = replace(seals[2], sealed_at="2026-10-26T00:00:00.000000Z")

    assert _check(sealed, seals=seals)[2] == (
        f"REJECT {D2}: the signature does not verify under the pinned key"
    )


def test_r6_a_reorder_in_storage_is_accepted(sealed):
    rows = _rows(sealed.copy, D0)
    _write_rows(sealed.copy, D0, rows[::-1])

    assert [r["key"] for r in _rows(sealed.copy, D0)] == ["sp:alice:c", "sp:alice:b", "sp:alice:a"]
    assert _check(sealed)[0] == f"ACCEPT {D0} n=3 head={sealed.seals[0].head.head}"


def test_a_seal_under_another_schema_is_rejected(sealed):
    lines = _check(sealed, schema="0" * 64)

    reason = f"sealed under schema {archive.HEAD_SCHEMA[:16]}…, not {'0' * 16}…"
    assert all(line.startswith("REJECT") and reason in line for line in lines)


def test_an_export_that_cannot_be_read_is_rejected(sealed):
    _export(sealed.copy, D2).write_bytes(b"not parquet")

    assert _check(sealed)[2].startswith(f"REJECT {D2}: export unreadable: ")


def test_days_the_copy_holds_without_a_seal_are_listed(sealed):
    assert verify.unsealed_days(sealed.copy, sealed.seals) == []
    assert verify.unsealed_days(sealed.copy, sealed.seals[:1]) == [D1.isoformat(), D2.isoformat()]


def test_a_seal_file_named_for_another_day_is_refused(sealed):
    directory = sealed.collector.root / collector.SEALED_DIR
    (directory / f"{D0}.json").rename(directory / "2026-01-01.json")

    with pytest.raises(ValueError, match="holds the seal of"):
        verify.load_seals(directory)


# --- the command lines ------------------------------------------------------------------------


def test_the_command_lines_seal_and_verify(sealed, tmp_path, capsys):
    root = str(sealed.collector.root)
    assert collector.main([root, "pub"]) == 0
    pub = capsys.readouterr().out.strip()
    assert pub == sealed.collector.public_key

    heads = [str(_head_file(sealed.archive, day)) for day in (D0, D1, D2)]
    assert collector.main([root, "seal", *heads]) == 0
    out = capsys.readouterr().out.splitlines()
    assert [chain.SealedHead.from_json(line) for line in out] == sealed.seals

    shutil.copytree(sealed.copy / D2.isoformat(), sealed.copy / D3.isoformat())
    args = [str(sealed.copy), str(sealed.collector.root / collector.SEALED_DIR), "--pub", pub]
    assert verify.main(args) == 0
    assert capsys.readouterr().out.splitlines() == [*_check(sealed), f"UNSEALED {D3}"]

    _write_rows(sealed.copy, D2, _rows(sealed.copy, D2)[:1])
    assert verify.main(args) == 1
    assert capsys.readouterr().out.splitlines()[2].startswith(f"REJECT {D2}: 1 rows, sealed 2")

    # No directory, and one that holds an unsigned head where seals belong.
    unsigned = sealed.copy / D0.isoformat() / archive.EXPORT_SUBDIR
    for sealed_dir in (tmp_path / "absent", unsigned):
        with pytest.raises(SystemExit) as exited:
            verify.main([str(sealed.copy), str(sealed_dir), "--pub", pub])
        assert exited.value.code == 2
    capsys.readouterr()

    doctored = tmp_path / "doctored.json"
    doctored.write_text(replace(sealed.seals[2].head, n=1).to_json())
    assert collector.main([root, "seal", str(doctored)]) == 1
    assert capsys.readouterr().err.startswith(f"refused {doctored}: {D2} is sealed with head ")
