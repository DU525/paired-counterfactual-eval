"""Crash-aware artifact bundle writer for paired-state experiments.

Design contract
---------------
* Callers hand over **already-serialized bytes**; this module never imports a
  simulator, so the runtime adapter stays separately reviewable.
* Every artifact is written with a write-temp / fsync / atomic-replace
  sequence. A crash can leave a ``.part`` file behind, but never a
  half-written target.
* ``RUN_STATE.json`` transitions to ``COMPLETE`` in one atomic replace. That
  single write is the commit point: until it happens the bundle is
  ``INCOMPLETE`` and must be rejected by any analysis.
* ``MANIFEST.json`` records size and SHA256 of every artifact, and
  ``verify_committed_bundle`` re-checks them all on read.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable, Optional

_SAFE_RUN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_RESERVED = {"RUN_STATE.json", "MANIFEST.json"}


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsync_dir(directory: Path) -> None:
    """POSIX directory fsync makes a rename durable.

    Windows does not expose the same operation through Python; file contents
    are still fsynced there, the weaker-but-acceptable guarantee.
    """
    if os.name != "posix":
        return
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        with temp.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        _fsync_dir(path.parent)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


class BundleTransaction:
    """Persist named byte artifacts; ``RUN_STATE=COMPLETE`` is the commit point."""

    def __init__(
        self,
        root: "str | Path",
        run_id: str,
        fault_hook: Optional[Callable[[str, Optional[str]], None]] = None,
    ) -> None:
        if not _SAFE_RUN.fullmatch(run_id):
            raise ValueError("run_id must be a safe, bounded identifier")
        self.root = Path(root).resolve() / run_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.run_id = run_id
        self._fault_hook = fault_hook
        self._entries: "dict[str, dict[str, object]]" = {}
        self._write_state("INCOMPLETE", reason="transaction_open")

    def _hook(self, stage: str, name: Optional[str] = None) -> None:
        if self._fault_hook is not None:
            self._fault_hook(stage, name)

    def _write_state(self, status: str, **extra: object) -> None:
        _atomic_replace(
            self.root / "RUN_STATE.json",
            _canonical_json({"run_id": self.run_id, "status": status, **extra}),
        )

    def write_bytes(self, name: str, payload: bytes) -> "dict[str, object]":
        rel = Path(name)
        if rel.is_absolute() or not rel.parts or any(p in ("", ".", "..") for p in rel.parts):
            raise ValueError("artifact name must be a safe relative path")
        if name in _RESERVED or name in self._entries:
            raise ValueError("reserved or duplicate artifact name")
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes serialized by the caller")
        target = (self.root / rel).resolve()
        if self.root not in target.parents:
            raise ValueError("artifact path escapes transaction root")
        _atomic_replace(target, payload)
        entry = {"name": rel.as_posix(), "bytes": len(payload), "sha256": _sha(payload)}
        self._entries[name] = entry
        self._hook("after_artifact_replace", name)
        manifest = {
            "run_id": self.run_id,
            "status": "INCOMPLETE",
            "artifacts": [self._entries[k] for k in sorted(self._entries)],
        }
        _atomic_replace(self.root / "MANIFEST.json", _canonical_json(manifest))
        self._hook("after_manifest_replace", name)
        return entry

    def commit(self) -> Path:
        if not self._entries:
            raise RuntimeError("cannot commit an empty bundle")
        artifacts = [self._entries[k] for k in sorted(self._entries)]
        for item in artifacts:
            path = self.root / str(item["name"])
            data = path.read_bytes()
            if len(data) != item["bytes"] or _sha(data) != item["sha256"]:
                raise RuntimeError(f"artifact verification failed: {item['name']}")
        manifest = {"run_id": self.run_id, "status": "COMPLETE", "artifacts": artifacts}
        manifest_bytes = _canonical_json(manifest)
        _atomic_replace(self.root / "MANIFEST.json", manifest_bytes)
        self._hook("before_commit_marker", None)
        # This atomic state replacement is the only commit point. A crash before
        # it leaves INCOMPLETE, which is inadmissible for analysis.
        self._write_state(
            "COMPLETE",
            manifest_sha256=_sha(manifest_bytes),
            artifact_count=len(artifacts),
        )
        self._hook("after_commit_marker", None)
        return self.root


def verify_committed_bundle(root: "str | Path") -> "dict[str, object]":
    """Re-verify a committed bundle. Raises on any inconsistency."""
    root = Path(root).resolve()
    state = json.loads((root / "RUN_STATE.json").read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETE":
        raise RuntimeError("bundle is not committed")
    manifest_path = root / "MANIFEST.json"
    manifest_bytes = manifest_path.read_bytes()
    if _sha(manifest_bytes) != state.get("manifest_sha256"):
        raise RuntimeError("manifest digest mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "COMPLETE" or manifest.get("run_id") != state.get("run_id"):
        raise RuntimeError("manifest identity/status mismatch")
    seen: "set[str]" = set()
    for item in manifest.get("artifacts", []):
        name = str(item["name"])
        rel = Path(name)
        if rel.is_absolute() or any(p in ("", ".", "..") for p in rel.parts) or name in seen:
            raise RuntimeError("unsafe or duplicate manifest path")
        seen.add(name)
        path = (root / rel).resolve()
        if root not in path.parents:
            raise RuntimeError("manifest path escapes bundle")
        data = path.read_bytes()
        if len(data) != item["bytes"] or _sha(data) != item["sha256"]:
            raise RuntimeError(f"artifact digest mismatch: {name}")
    if len(seen) != int(state.get("artifact_count", -1)):
        raise RuntimeError("committed artifact count mismatch")
    return manifest