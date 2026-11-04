"""The day chain: leaves, the fold, day heads and the bytes a collector signs.

The hex strings are the vectors of the record-chain design, computed from its
synthetic day: four invented entries in the stored shape of the export. They
pin the canonical form, so a change to any of them is a change of format, not
a refactor.
"""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from osprey.agent_record import archive, chain
from osprey.agent_record import contract as c

DAY = "2026-11-13"
DAY_BEFORE = "2026-11-12"
DAY_AFTER = "2026-11-14"


def _compact(body: dict) -> str:
    return json.dumps(body, separators=(",", ":"))


def _row(key: str, kind: str, ts: str, body: dict, inserted_at: str, **fields: str) -> dict:
    row = {
        "key": key,
        "chunk_index": 0,
        "chunk_total": 1,
        "kind": kind,
        "ts": ts,
        "identity": "demo",
        "on_behalf_of": "",
        "session_id": "s1",
        "terminal_session": "",
        "run_id": "",
        "agent_id": "",
        "parent_key": "",
        "tool_use_id": "",
        "body": _compact(body),
        "inserted_at": inserted_at,
    }
    row.update(fields)
    return row


#: The design's synthetic day, in export order; times already in their wire form.
FIXTURE = [
    _row(
        "cc:demo:s1:u1:0",
        "prompt",
        "2026-11-13T09:00:00.100000Z",
        {"text": "set HCM05 to 0.25"},
        "2026-11-13T09:00:00.101000Z",
    ),
    _row(
        "cc:demo:s1:u2:0",
        "tool_call",
        "2026-11-13T09:00:01.000000Z",
        {"server": "bluesky", "tool": "queue_add"},
        "2026-11-13T09:00:01.002000Z",
        tool_use_id="tu_7",
    ),
    _row(
        "sp:demo:a1",
        "audit",
        "2026-11-13T09:00:01.050000Z",
        {
            "surface": "hook_approval",
            "decision": "approved",
            "detail": "approver=write_switch position=allowed",
        },
        "2026-11-13T09:00:01.060000Z",
        tool_use_id="tu_7",
    ),
    _row(
        "cc:demo:s1:u3:0",
        "tool_result",
        "2026-11-13T09:00:01.400000Z",
        {"ok": True, "channel": "SR:HCM05", "value": 0.25},
        "2026-11-13T09:00:01.402000Z",
        tool_use_id="tu_7",
    ),
]

ACCEPT = "33dec3a6d93ce916dae2436235ad9739573a319ce3d87c884e471bd0a3429b79"
R1_EDIT = "b1a33f3269307f19aa30fa59985113e40e7b0629365ef8a2524fb1b3f1f70b24"
R2_DELETE = "b324102a2bf63722777e6b49ff86128e10d932cfd528ea1dc3b0b36fc7e7ff43"
R3_INSERT = "d7c2f2a4a29e235387b48da70cf6bf80cf266fdd18ce5b7c870efbfc9058a8c5"
BEFORE_HEAD = "c0352cd34b9d449b286b4a39c6ac6a7c4a31f156f1c1eb72ce897cc9642dcb06"
CHAINED = "55e6a5c20af11044c0f3dbd56f742db163edc34c2585f577ed06cf761e158958"
R4C_EDITED_BEFORE = "bdffa68cdf81d0a410bc64406929f12278dfbf4bd2afa7abbd662675dc360570"
A2_IDLE = "16788748c3b914570d4cf1fd5ca931e5eb799f49fea31b7177b0e363a0f8aebd"


def _head(day: str, rows, prev: bytes = chain.ZERO_HEAD) -> str:
    return chain.fold(day, rows, prev).head.hex()


# --- the design's vectors ---------------------------------------------------------------------


def test_the_design_day_folds_to_its_head():
    folded = chain.fold(DAY, FIXTURE)
    assert (folded.head.hex(), folded.n) == (ACCEPT, 4)
    assert folded.leaves == tuple(chain.leaf(row) for row in FIXTURE)


def test_r1_an_edited_field_moves_the_head():
    edited = [dict(row) for row in FIXTURE]
    body = json.loads(edited[3]["body"])
    body["value"] = 0.05
    edited[3]["body"] = _compact(body)
    assert _head(DAY, edited) == R1_EDIT


def test_r2_a_deleted_row_moves_the_head():
    assert _head(DAY, [row for row in FIXTURE if row["key"] != "sp:demo:a1"]) == R2_DELETE


def test_r3_a_back_dated_row_moves_the_head():
    extra = {
        **FIXTURE[2],
        "key": "sp:demo:a9",
        "ts": "2026-11-13T09:00:00.500000Z",
        "inserted_at": "2026-11-13T09:00:00.500000Z",
    }
    assert _head(DAY, [*FIXTURE, extra]) == R3_INSERT


def test_r6_a_reorder_in_storage_is_not_a_change():
    assert _head(DAY, [FIXTURE[2], FIXTURE[0], FIXTURE[3], FIXTURE[1]]) == ACCEPT


def test_a_day_folds_from_the_head_of_the_day_before():
    before = chain.fold(DAY_BEFORE, FIXTURE[:2]).head
    assert before.hex() == BEFORE_HEAD
    assert _head(DAY, FIXTURE, before) == CHAINED


def test_r4b_a_day_whose_day_before_is_gone_cannot_pass_as_a_first_day():
    # From zeros the day folds to its first-day head, never to the head it
    # was sealed with.
    assert _head(DAY, FIXTURE, chain.ZERO_HEAD) == ACCEPT != CHAINED


def test_r4c_an_edited_day_before_moves_the_genesis_of_the_day_after():
    edited = [dict(row) for row in FIXTURE[:2]]
    edited[0]["body"] += "x"
    assert _head(DAY, FIXTURE, chain.fold(DAY_BEFORE, edited).head) == R4C_EDITED_BEFORE


def test_a2_an_idle_day_is_its_genesis_and_still_chains():
    folded = chain.fold(DAY_AFTER, [], bytes.fromhex(CHAINED))
    assert (folded.head.hex(), folded.n, folded.leaves) == (A2_IDLE, 0, ())
    assert folded.head == chain.genesis(DAY_AFTER, bytes.fromhex(CHAINED))


def test_length_prefixes_keep_adjacent_columns_apart():
    joined = {**FIXTURE[0], "session_id": "ab", "terminal_session": ""}
    split = {**FIXTURE[0], "session_id": "a", "terminal_session": "b"}
    assert chain.leaf(joined) != chain.leaf(split)


# --- the canonical form -----------------------------------------------------------------------


def test_a_leaf_covers_every_stored_column():
    assert set(FIXTURE[0]) == set(c.ENTRY_COLUMNS)
    for column in c.ENTRY_COLUMNS:
        changed = dict(FIXTURE[0])
        if column in ("chunk_index", "chunk_total"):
            changed[column] += 1
        elif column in ("ts", "inserted_at"):
            changed[column] = "2026-11-13T09:00:00.100001Z"
        else:
            changed[column] += "x"
        assert chain.leaf(changed) != chain.leaf(FIXTURE[0]), column


def test_aware_times_hash_as_their_utc_wire_form():
    plus_two = timezone(timedelta(hours=2))
    row = {
        **FIXTURE[0],
        "ts": datetime(2026, 11, 13, 11, 0, 0, 100000, tzinfo=plus_two),
        "inserted_at": datetime(2026, 11, 13, 9, 0, 0, 101000, tzinfo=UTC),
    }
    assert chain.leaf(row) == chain.leaf(FIXTURE[0])


def test_the_wire_form_pads_every_field():
    assert chain.wire_time(datetime(1, 2, 3, 4, 5, 6, 7, tzinfo=UTC)) == (
        "0001-02-03T04:05:06.000007Z"
    )


@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 11, 13, 9, 0),  # naive
        "2026-11-13T09:00:00Z",  # no microseconds
        "2026-11-13 09:00:00.000000Z",
        "2026-11-13T09:00:00.000000+00:00",
    ],
)
def test_a_time_in_any_other_form_is_refused(value):
    with pytest.raises(ValueError):
        chain.leaf({**FIXTURE[0], "ts": value})


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("on_behalf_of", None),
        ("body", None),
        ("chunk_index", None),
        ("chunk_index", "0"),
        ("chunk_total", True),
        ("chunk_index", -1),
        ("body", b"{}"),
        ("identity", 7),
    ],
)
def test_a_value_without_a_canonical_form_is_refused_not_hashed(column, value):
    with pytest.raises(ValueError, match=column):
        chain.leaf({**FIXTURE[0], column: value})


def test_a_row_without_every_column_is_refused():
    row = dict(FIXTURE[0])
    del row["tool_use_id"]
    with pytest.raises(KeyError):
        chain.leaf(row)


@pytest.mark.parametrize(("day", "prev"), [("2026-11-1", bytes(32)), (DAY, bytes(31))])
def test_a_genesis_needs_a_day_and_a_32_byte_head(day, prev):
    with pytest.raises(ValueError):
        chain.genesis(day, prev)


# --- heads and seals --------------------------------------------------------------------------


def _day_head() -> chain.DayHead:
    return chain.DayHead(day=DAY, schema=archive.HEAD_SCHEMA, n=4, prev=BEFORE_HEAD, head=CHAINED)


def test_a_day_head_is_one_json_line_in_record_order():
    head = _day_head()
    line = head.to_json()
    assert line.endswith("\n") and line.count("\n") == 1
    assert list(json.loads(line)) == ["v", "day", "schema", "n", "prev", "head"]
    assert chain.DayHead.from_json(line) == head


@pytest.mark.parametrize(
    "change",
    [
        {"v": 2},
        {"day": "2026-11-31"},
        {"day": "20261113"},
        {"schema": archive.HEAD_SCHEMA.upper()},
        {"n": -1},
        {"n": True},
        {"prev": BEFORE_HEAD[:-2]},
        {"head": "z" * 64},
    ],
)
def test_a_malformed_day_head_is_refused(change):
    with pytest.raises(ValueError):
        replace(_day_head(), **change)


@pytest.mark.parametrize("text", ["[]", '{"v": 1}', '{"day": null}'])
def test_text_that_is_not_a_day_head_is_refused(text):
    with pytest.raises(ValueError):
        chain.DayHead.from_json(text)


def test_the_seal_message_binds_every_field_and_the_sealing_time():
    head = _day_head()
    at = "2026-11-14T00:05:00.000000Z"
    message = chain.seal_message(head, at)
    assert message.startswith(chain.SEAL_TAG)
    changed = [
        chain.seal_message(replace(head, n=5), at),
        chain.seal_message(replace(head, day=DAY_AFTER), at),
        chain.seal_message(replace(head, prev=CHAINED), at),
        chain.seal_message(replace(head, head=BEFORE_HEAD), at),
        chain.seal_message(replace(head, schema=CHAINED), at),
        chain.seal_message(head, "2026-11-14T00:05:00.000001Z"),
    ]
    assert len({message, *changed}) == 1 + len(changed)


def test_a_sealed_head_round_trips_and_refuses_a_malformed_signature():
    sealed = chain.SealedHead(
        head=_day_head(), sealed_at="2026-11-14T00:05:00.000000Z", sig="ab" * 64, pub="cd" * 32
    )
    record = json.loads(sealed.to_json())
    assert list(record)[-3:] == ["sealed_at", "sig", "pub"]
    assert chain.SealedHead.from_json(sealed.to_json()) == sealed
    with pytest.raises(ValueError):
        replace(sealed, sig="ab" * 63)
    with pytest.raises(ValueError):
        replace(sealed, sealed_at="2026-11-14T00:05:00Z")


# --- what the host imports --------------------------------------------------------------------


def _imports(module) -> set[str]:
    """Every module *module* imports, and every name it imports from one, dotted."""
    names = set()
    for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_the_chain_is_standard_library_only_and_does_no_io():
    names = _imports(chain)
    roots = {name.split(".")[0] for name in names} - {"__future__"}
    assert {r for r in roots if r not in sys.stdlib_module_names} == {"osprey"}
    assert {name for name in names if name.startswith("osprey.")} == {
        "osprey.agent_record.contract",
        "osprey.agent_record.contract.ENTRY_COLUMNS",
    }
    assert not {"os", "io", "shutil", "subprocess", "socket", "pathlib"} & roots


def test_the_host_side_holds_no_signing_code():
    for module in (chain, archive):
        names = _imports(module)
        assert not {n for n in names if n.split(".")[0] in {"cryptography", "nacl"}}, module
        assert not {"osprey.agent_record.collector", "osprey.agent_record.verify"} & names
