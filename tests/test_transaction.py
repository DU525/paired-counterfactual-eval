"""Crash-safety contract of the artifact bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcf_eval.core.transaction import BundleTransaction, verify_committed_bundle


def test_commit_makes_bundle_verifiable(tmp_path: Path):
    tx = BundleTransaction(tmp_path, "run-001")
    tx.write_bytes("a.json", b'{"x":1}')
    tx.write_bytes("nested/b.bin", b"\x00\x01\x02")
    root = tx.commit()
    manifest = verify_committed_bundle(root)
    assert manifest["status"] == "COMPLETE"
    assert len(manifest["artifacts"]) == 2


def test_uncommitted_bundle_is_rejected(tmp_path: Path):
    tx = BundleTransaction(tmp_path, "run-002")
    tx.write_bytes("a.json", b"{}")
    with pytest.raises(RuntimeError, match="not committed"):
        verify_committed_bundle(tx.root)


def test_tampered_artifact_is_detected(tmp_path: Path):
    tx = BundleTransaction(tmp_path, "run-003")
    tx.write_bytes("a.json", b'{"x":1}')
    root = tx.commit()
    (root / "a.json").write_bytes(b'{"x":2}')
    with pytest.raises(RuntimeError, match="digest mismatch"):
        verify_committed_bundle(root)


def test_crash_before_commit_leaves_incomplete(tmp_path: Path):
    def crash(stage: str, name: str | None) -> None:
        if stage == "before_commit_marker":
            raise RuntimeError("simulated crash")

    tx = BundleTransaction(tmp_path, "run-004", fault_hook=crash)
    tx.write_bytes("a.json", b"{}")
    with pytest.raises(RuntimeError, match="simulated crash"):
        tx.commit()
    state = json.loads((tx.root / "RUN_STATE.json").read_text(encoding="utf-8"))
    assert state["status"] == "INCOMPLETE"
    with pytest.raises(RuntimeError):
        verify_committed_bundle(tx.root)


def test_path_traversal_is_rejected(tmp_path: Path):
    tx = BundleTransaction(tmp_path, "run-005")
    with pytest.raises(ValueError):
        tx.write_bytes("../escape.json", b"{}")
    with pytest.raises(ValueError):
        tx.write_bytes("/abs.json", b"{}")
    with pytest.raises(ValueError):
        tx.write_bytes("RUN_STATE.json", b"{}")


def test_run_id_is_sanitised(tmp_path: Path):
    with pytest.raises(ValueError):
        BundleTransaction(tmp_path, "../evil")