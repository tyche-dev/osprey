"""The reference collector — off the host, the only signing key, one seal per day.

The host writes each exported day's head unsigned beside the export
(:data:`~osprey.agent_record.chain.HEAD_FILE`) and never holds a key. The
deployment ships that file here, on whatever path the facility chooses; this
process runs where the deployment cannot reach back into it, and the key
never leaves its directory.

The collector keeps one directory:

- ``signing-key.pem`` — its Ed25519 key (PKCS#8, unencrypted, mode ``0600``),
  created the first time it runs. That key is a **test key**: it is good for
  evaluating the chain and for the tests, and a facility seals with a key the
  lab holds and manages, not with one this module made.
- ``sealed/<day>.json`` — one :class:`~osprey.agent_record.chain.SealedHead`
  per day, never rewritten.

**First seal wins.** The first head received for a day is signed (Ed25519 over
:func:`~osprey.agent_record.chain.seal_message`) and kept. A second, different
head for that day is refused (:class:`SealRefused`), so a host that later
recomputes a day over doctored rows cannot get it sealed. The same head again
returns the seal already kept.

Usage::

    python -m osprey.agent_record.collector DIR seal HEAD_FILE...
    python -m osprey.agent_record.collector DIR pub

``seal`` prints each sealed head as one JSON line and exits 1 if any head is
refused; ``pub`` prints the public key a verifier pins.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from osprey.agent_record import chain

__all__ = ["KEY_FILE", "SEALED_DIR", "Collector", "SealRefused", "main"]

KEY_FILE = "signing-key.pem"
SEALED_DIR = "sealed"


class SealRefused(Exception):
    """A head for a day already sealed with a different head."""


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new(path: Path, data: bytes, mode: int) -> bool:
    """Write *data* to *path*, mode *mode* whatever the umask, only if nothing is there yet.

    Staged beside *path* and linked into place, so *path* never holds a part
    of *data* and of two writers exactly one wins; the directory is synced
    once *path* is in it, so a seal returned outlives a crash. Returns whether
    this call wrote it.
    """
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        try:
            os.fchmod(fd, mode)
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(temp, path)
        except FileExistsError:
            return False
        _fsync_dir(path.parent)
        return True
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


def _load_or_create_key(path: Path) -> Ed25519PrivateKey:
    if not path.exists():
        pem = Ed25519PrivateKey.generate().private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        _write_new(path, pem, 0o600)
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"{path} does not hold an Ed25519 key")
    return key


class Collector:
    """One collector directory: its key and the heads it sealed.

    Args:
        root: The collector's directory, created (mode ``0700``) when missing,
            with a new test key on first use.
        clock: Where a seal's ``sealed_at`` comes from; the UTC wall clock
            unless given.
    """

    def __init__(self, root: Path, clock: Callable[[], datetime] | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        (self.root / SEALED_DIR).mkdir(mode=0o700, exist_ok=True)
        self._key = _load_or_create_key(self.root / KEY_FILE)
        self._clock = clock or (lambda: datetime.now(UTC))
        self.public_key = (
            self._key.public_key()
            .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            .hex()
        )

    def _path(self, day: str) -> Path:
        return self.root / SEALED_DIR / f"{date.fromisoformat(day).isoformat()}.json"

    def sealed(self, day: str) -> chain.SealedHead | None:
        """The seal kept for *day*, or ``None``."""
        try:
            return chain.SealedHead.from_json(self._path(day).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def seal(self, head: chain.DayHead) -> chain.SealedHead:
        """Seal *head* if its day has no seal yet; the seal kept for its day.

        Raises:
            SealRefused: If the day is already sealed with a different head.
        """
        kept = self.sealed(head.day)
        if kept is None:
            sealed_at = chain.wire_time(self._clock())
            sig = self._key.sign(chain.seal_message(head, sealed_at)).hex()
            new = chain.SealedHead(head=head, sealed_at=sealed_at, sig=sig, pub=self.public_key)
            if _write_new(self._path(head.day), new.to_json().encode("ascii"), 0o644):
                return new
            kept = self.sealed(head.day)
            if kept is None:
                raise RuntimeError(f"the seal of {head.day} was written and is gone")
        if kept.head != head:
            raise SealRefused(
                f"{head.day} is sealed with head {kept.head.head} (n={kept.head.n}); "
                f"refusing head {head.head} (n={head.n})"
            )
        return kept


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m osprey.agent_record.collector",
        description="Seal day heads off the host: first seal wins.",
    )
    parser.add_argument("root", type=Path, help="the collector's directory")
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal", help="seal each day head file given")
    seal.add_argument("heads", nargs="+", type=Path)
    commands.add_parser("pub", help="print the public key a verifier pins")
    args = parser.parse_args(argv)

    try:
        collector = Collector(args.root)
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "pub":
        sys.stdout.write(collector.public_key + "\n")
        return 0
    refused = 0
    for path in args.heads:
        try:
            sealed = collector.seal(chain.DayHead.from_json(path.read_text(encoding="utf-8")))
        except (SealRefused, ValueError, OSError) as exc:
            sys.stderr.write(f"refused {path}: {exc}\n")
            refused += 1
            continue
        sys.stdout.write(sealed.to_json())
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
