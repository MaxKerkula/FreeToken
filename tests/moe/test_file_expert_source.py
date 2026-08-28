from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from uuid import uuid4

import pytest
import torch

from freetoken.moe.expert_source import (
    ExpertSourceError,
    FileExpertSource,
    PLANE_LAYOUT,
    RAW_RECORD_BYTES,
)


@pytest.fixture
def z_fixture_dir():
    root = Path.cwd() / ".stage6-test-fixtures" / uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    try:
        assert (root.drive or "").upper() == "Z:", root
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _record(byte: int) -> bytes:
    return bytes([byte]) * RAW_RECORD_BYTES


def _resident_banks(num_layers: int, experts: int):
    shapes = {
        "gate_up_packed": ((experts, 1280, 1280), torch.uint8),
        "gate_up_scale": ((experts, 1280, 160), torch.float8_e4m3fn),
        "gate_up_global": ((experts, 1280), torch.float16),
        "down_packed": ((experts, 2560, 320), torch.uint8),
        "down_scale": ((experts, 2560, 40), torch.float8_e4m3fn),
        "down_global": ((experts, 2560), torch.float16),
    }
    return {
        name: [torch.zeros(shape, dtype=dtype) if layer == 0 else None for layer in range(num_layers)]
        for name, (shape, dtype) in shapes.items()
    }


def test_file_expert_source_reads_exact_planes_and_rejects_tamper(z_fixture_dir):
    root = z_fixture_dir
    path = root / "experts-L00.nvfp4"
    digest = FileExpertSource.create_synthetic(path, num_experts=3, records=[_record(i) for i in range(3)])
    with FileExpertSource(path, num_experts=3, expected_sha256=digest) as src:
        rows = src.read_record(2)
        assert set(rows) == {name for name, _, _ in PLANE_LAYOUT}
        assert rows["gate_up_packed"].shape == (1280, 1280)
        assert rows["gate_up_scale"].shape == (1280, 160)
        assert int(rows["gate_up_packed"].flatten()[0]) == 2
        assert src.read_count == 1
        assert src.max_inflight <= 1 <= 16
    with pytest.raises(ExpertSourceError, match="closed"):
        src.read_record(0)


def test_file_expert_source_cache_miss_fills_slots_without_host_layer(z_fixture_dir):
    path = z_fixture_dir / "experts-L01.nvfp4"
    digest = FileExpertSource.create_synthetic(path, num_experts=3, records=[_record(i + 11) for i in range(3)])
    src = FileExpertSource(path, num_experts=3, expected_sha256=digest)
    try:
        from freetoken.moe.offload_cache import OffloadMoeCache

        cache = OffloadMoeCache(
            num_layers=2,
            num_experts=3,
            cache_size=3,
            device=torch.device("cpu"),
            quant_format="nvfp4",
            decode_target="gpu",
        )
        cache.set_bank_sources(_resident_banks(2, 3))
        cache.set_file_sources({1: src})
        assert all(cache.bank_sources[name][1] is None for name in cache.bank_schema)
        ids = torch.tensor([2, 0], dtype=torch.int32)
        cache.ensure_experts(1, ids)
        assert ids.tolist() == [0, 1]
        cache.copy_missing()
        assert cache.bank_caches["gate_up_packed"][0, 0, 0].item() == 13
        assert cache.bank_caches["gate_up_packed"][1, 0, 0].item() == 11
        assert src.bytes_read == 2 * src.record_bytes
        assert cache._pending_file_fetches == []
        cache.reset()
        assert cache.id_of_slot.tolist() == [-1, -1, -1]
        assert cache.slot_for_id.tolist() == [[-1, -1, -1], [-1, -1, -1]]
    finally:
        src.close()


def test_file_tier_materialize_streams_complete_layer(z_fixture_dir):
    path = z_fixture_dir / "experts-L01.nvfp4"
    digest = FileExpertSource.create_synthetic(path, num_experts=3, records=[_record(i + 21) for i in range(3)])
    src = FileExpertSource(path, num_experts=3, expected_sha256=digest)
    try:
        from freetoken.moe.offload_cache import OffloadMoeCache

        cache = OffloadMoeCache(2, 3, 3, torch.device("cpu"), quant_format="nvfp4")
        cache.set_bank_sources(_resident_banks(2, 3))
        cache.set_file_sources({1: src})
        cache.materialize_layer(1)
        cache.copy_missing()
        values = cache.bank_caches["gate_up_packed"][:, 0, 0].tolist()
        assert values == [21, 22, 23]
        assert src.read_count == 3
    finally:
        src.close()


def test_file_tier_payload_hash_fails_closed(z_fixture_dir):
    path = z_fixture_dir / "experts-L00.nvfp4"
    FileExpertSource.create_synthetic(path, num_experts=1, records=[_record(9)])
    with path.open("r+b") as fh:
        fh.seek(4096 + 17)
        fh.write(b"x")
    with pytest.raises(ExpertSourceError, match="payload hash mismatch"):
        FileExpertSource(path, num_experts=1)


def test_file_tier_rejects_cpu_hybrid_and_overlap(z_fixture_dir):
    path = z_fixture_dir / "experts-L00.nvfp4"
    digest = FileExpertSource.create_synthetic(path, num_experts=1, records=[_record(7)])
    src = FileExpertSource(path, num_experts=1, expected_sha256=digest)
    try:
        from freetoken.moe.offload_cache import OffloadMoeCache

        for target in ("cpu", "hybrid"):
            cache = OffloadMoeCache(1, 1, 1, torch.device("cpu"), decode_target=target, quant_format="nvfp4")
            # Source registration itself fails before any source can be used;
            # a direct call is sufficient to prove the policy and avoids giant
            # synthetic resident allocations in this negative test.
            with pytest.raises(ValueError, match="GPU-only"):
                cache.set_file_sources({0: src})
    finally:
        src.close()


def test_file_expert_source_rejects_wrong_volume(monkeypatch):
    with pytest.raises(ExpertSourceError, match="Z:"):
        FileExpertSource(r"C:\\not-a-tier.nvfp4")
