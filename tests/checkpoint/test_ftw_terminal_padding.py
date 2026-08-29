from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import freetoken.checkpoint.ftw as ftw_module
import freetoken.checkpoint.step9b_executor as executor_module
from freetoken.checkpoint.ftw import (
    FTWReader,
    FTWWriter,
    ensure_ftw_terminal_padding,
    iter_ftw_weights,
)
from freetoken.checkpoint.step9b_executor import Step9BExecutor


ALIGN = 4096


def _fixture(root: Path, *, shard_limit: int = 1 << 20) -> tuple[Path, dict, bytes]:
    root.mkdir()
    writer = FTWWriter(str(root), shard_limit=shard_limit)
    tensors = {
        "first": torch.arange(32, dtype=torch.int32),
        "second": torch.arange(15, dtype=torch.float32),
        "third": torch.tensor([3, 1, 4, 1, 5], dtype=torch.bfloat16),
    }
    for name, tensor in tensors.items():
        writer.add_tensor(name, tensor)
    index = writer.finalize({"source_inventory_sha256": "a" * 64})
    shard = root / index["shards"][-1]["file"]
    return root, index, shard.read_bytes()


def _index(root: Path) -> dict:
    return json.loads((root / "freetoken_weight.json").read_text(encoding="utf-8"))


def test_terminal_padding_preserves_tensor_inventory_prefix_and_reader(tmp_path: Path) -> None:
    root, before, before_bytes = _fixture(tmp_path / "active")
    expected = int(before["total_bytes"])
    target = expected + ALIGN
    before_tensors = before["tensors"]
    before_prefix = before_bytes

    result = ensure_ftw_terminal_padding(
        str(root), expected_unpadded_bytes=expected, target_bytes=target
    )

    after = _index(root)
    shard = root / after["shards"][-1]["file"]
    after_bytes = shard.read_bytes()
    assert result["state"] == "PADDED"
    assert after["total_bytes"] == target
    assert after["tensors"] == before_tensors
    assert after["shards"][:-1] == before["shards"][:-1]
    assert after["shards"][-1]["global_off"] == before["shards"][-1]["global_off"]
    assert after["shards"][-1]["nbytes"] == before["shards"][-1]["nbytes"] + ALIGN
    assert after_bytes[: len(before_prefix)] == before_prefix
    assert after_bytes[len(before_prefix) :] == b"\0" * ALIGN

    loaded = dict(iter_ftw_weights(str(root), workers=1))
    assert torch.equal(loaded["first"], torch.arange(32, dtype=torch.int32))
    assert torch.equal(loaded["second"], torch.arange(15, dtype=torch.float32))
    assert torch.equal(loaded["third"], torch.tensor([3, 1, 4, 1, 5], dtype=torch.bfloat16))
    reader = FTWReader(str(root))
    assert [entry["name"] for entry in reader.entries()] == ["first", "second", "third"]
    reader.close()


def test_terminal_padding_repeated_recovery_is_idempotent(tmp_path: Path) -> None:
    root, before, _ = _fixture(tmp_path / "active")
    expected = int(before["total_bytes"])
    target = expected + ALIGN
    first = ensure_ftw_terminal_padding(
        str(root), expected_unpadded_bytes=expected, target_bytes=target
    )
    first_digest = hashlib.sha256((root / "freetoken-00000.ftw").read_bytes()).hexdigest()
    second = ensure_ftw_terminal_padding(
        str(root), expected_unpadded_bytes=expected, target_bytes=target
    )
    second_digest = hashlib.sha256((root / "freetoken-00000.ftw").read_bytes()).hexdigest()
    assert first["state"] == "PADDED"
    assert second["state"] == "ALREADY_PADDED"
    assert first_digest == second_digest
    assert _index(root)["total_bytes"] == target


def test_terminal_padding_adopts_durable_orphan_tail(tmp_path: Path) -> None:
    root, before, _ = _fixture(tmp_path / "active")
    expected = int(before["total_bytes"])
    target = expected + ALIGN
    shard = root / before["shards"][-1]["file"]
    with shard.open("ab") as handle:
        handle.write(b"\0" * ALIGN)
        handle.flush()
    result = ensure_ftw_terminal_padding(
        str(root), expected_unpadded_bytes=expected, target_bytes=target
    )
    assert result["state"] == "RECOVERED"
    assert _index(root)["total_bytes"] == target
    assert shard.stat().st_size == target


@pytest.mark.parametrize("extra", [1, ALIGN - 1, ALIGN + 1, ALIGN * 2])
def test_terminal_padding_rejects_non_exact_or_oversize_tail(tmp_path: Path, extra: int) -> None:
    root, before, _ = _fixture(tmp_path / f"active-{extra}")
    expected = int(before["total_bytes"])
    shard = root / before["shards"][-1]["file"]
    with shard.open("ab") as handle:
        handle.write(b"\0" * extra)
    with pytest.raises(ValueError, match="extent discrepancy"):
        ensure_ftw_terminal_padding(
            str(root), expected_unpadded_bytes=expected, target_bytes=expected + ALIGN
        )
    assert _index(root)["total_bytes"] == expected


def test_terminal_padding_rejects_nonzero_orphan_tail(tmp_path: Path) -> None:
    root, before, _ = _fixture(tmp_path / "active")
    expected = int(before["total_bytes"])
    shard = root / before["shards"][-1]["file"]
    with shard.open("ab") as handle:
        handle.write(b"\0" * (ALIGN - 1) + b"x")
    with pytest.raises(ValueError, match="all-zero"):
        ensure_ftw_terminal_padding(
            str(root), expected_unpadded_bytes=expected, target_bytes=expected + ALIGN
        )
    assert _index(root)["total_bytes"] == expected


def test_terminal_padding_rejects_corrupt_target_tail(tmp_path: Path) -> None:
    root, before, _ = _fixture(tmp_path / "active")
    expected = int(before["total_bytes"])
    target = expected + ALIGN
    ensure_ftw_terminal_padding(str(root), expected_unpadded_bytes=expected, target_bytes=target)
    shard = root / _index(root)["shards"][-1]["file"]
    with shard.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"x")
    with pytest.raises(ValueError, match="all-zero"):
        ensure_ftw_terminal_padding(str(root), expected_unpadded_bytes=expected, target_bytes=target)


def test_terminal_padding_index_publication_failure_preserves_old_index_and_recovers(
    tmp_path: Path, monkeypatch
) -> None:
    root, before, _ = _fixture(tmp_path / "active")
    expected = int(before["total_bytes"])
    target = expected + ALIGN
    original_replace = ftw_module.os.replace

    def denied_replace(_source, _destination):
        error = OSError(5, "sharing/access lock")
        error.winerror = 5
        raise error

    monkeypatch.setattr(ftw_module.os, "replace", denied_replace)
    with pytest.raises(OSError, match="atomic replacement failed"):
        ensure_ftw_terminal_padding(
            str(root), expected_unpadded_bytes=expected, target_bytes=target
        )
    assert _index(root)["total_bytes"] == expected
    assert (root / "freetoken-00000.ftw").stat().st_size == target
    assert list(root.glob(".freetoken_weight.json.padding-*"))

    monkeypatch.setattr(ftw_module.os, "replace", original_replace)
    recovered = ensure_ftw_terminal_padding(
        str(root), expected_unpadded_bytes=expected, target_bytes=target
    )
    assert recovered["state"] == "RECOVERED"
    assert _index(root)["total_bytes"] == target
    assert not list(root.glob(".freetoken_weight.json.padding-*"))


def test_terminal_padding_rejects_conflicting_stale_index_temp(tmp_path: Path) -> None:
    root, before, _ = _fixture(tmp_path / "active")
    expected = int(before["total_bytes"])
    target = expected + ALIGN
    ensure_ftw_terminal_padding(str(root), expected_unpadded_bytes=expected, target_bytes=target)
    stale = root / ".freetoken_weight.json.padding-conflict"
    stale.write_text('{"format":"not-the-published-index"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting FTW terminal-padding temp"):
        ensure_ftw_terminal_padding(
            str(root), expected_unpadded_bytes=expected, target_bytes=target
        )
    assert stale.is_file()


def test_terminal_padding_conflict_preserves_all_stale_candidates(tmp_path: Path) -> None:
    root, before, _ = _fixture(tmp_path / "active")
    expected = int(before["total_bytes"])
    target = expected + ALIGN
    ensure_ftw_terminal_padding(str(root), expected_unpadded_bytes=expected, target_bytes=target)
    published = _index(root)
    valid = root / ".freetoken_weight.json.padding-aaa-valid"
    valid.write_text(json.dumps(published), encoding="utf-8")
    conflict = root / ".freetoken_weight.json.padding-zzz-conflict"
    conflict.write_text('{"format":"conflict"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting FTW terminal-padding temp"):
        ensure_ftw_terminal_padding(
            str(root), expected_unpadded_bytes=expected, target_bytes=target
        )
    assert valid.is_file()
    assert conflict.is_file()


def test_executor_b4_receipt_recovery_seam_pads_candidate(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "target"
    scratch = tmp_path / "scratch"
    base = ALIGN
    frozen = base + ALIGN

    def fake_convert(_source: str, out_dir: str, **_kwargs):
        active = Path(out_dir) / "qwen4-active-v1.ftw"
        writer = FTWWriter(str(active), shard_limit=1 << 20)
        writer.add_tensor("active.weight", torch.arange(8, dtype=torch.float32))
        index = writer.finalize({"source_inventory_sha256": "b" * 64})
        assert index["total_bytes"] == base
        return {"copied_metadata": []}

    monkeypatch.setattr(executor_module, "ACTIVE_BYTES", frozen)
    monkeypatch.setattr("freetoken.checkpoint.convert.convert_checkpoint", fake_convert)
    executor = object.__new__(Step9BExecutor)
    executor.execute = True
    executor.source_root = source
    executor.target_root = target
    executor.scratch_root = scratch
    executor.source_inventory_fingerprint = "b" * 64
    executor.builder_commit = "c" * 40
    executor.runtime_commit = "d" * 40
    executor.manifest = SimpleNamespace(revision="e" * 40, rows_for_stage=lambda _stage: ())
    executor._disk_gate = lambda: None
    executor._host_gate = lambda: None
    executor._source_bindings = lambda _rows: []

    result = Step9BExecutor.convert_and_validate_active(executor)
    assert result["state"] == "COMPLETE"
    assert result["target_bytes"] == frozen
    index = _index(target / "qwen4-active-v1.ftw")
    assert index["total_bytes"] == frozen
    receipt = json.loads((scratch / "receipts" / "B4-active.json").read_text())
    assert receipt["completion"] == "COMPONENT_COMPLETE"
    assert receipt["validation"]["terminal_padding"]["state"] == "PADDED"
