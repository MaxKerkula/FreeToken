from __future__ import annotations

import json
import math
import time
from types import SimpleNamespace

import pytest
import torch

import freetoken.attention.qsa as qsa_module
from freetoken.attention.qsa import QSAAttnBackend, select_qsa_logical_rows
from freetoken.distributed.info import DistributedInfo
from freetoken.kvcache.qsa_pool import QSAKVCache


SEED = 38038
RATIO = 4
BUDGET = 2048
LAYER_ID = 1
INDEX_DIM = 8


class _RecordingSyntheticIndexer:
    """Observable stand-in for official K RMSNorm + block-start RoPE.

    The backend owns pooling and position routing, while the model indexer owns
    normalization/RoPE. This deterministic transform makes both inputs visible
    without importing model weights or calling a production selection helper.
    """

    def __init__(self):
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def normalize_compressed_keys(
        self, keys: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append((keys.detach().cpu().clone(), positions.detach().cpu().clone()))
        values = keys.float()
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
        # A small block-start marker makes an incorrect current/end position observable.
        marker = positions[0].float().view(-1, 1, 1) / 4096.0
        return (normalized + marker).to(keys.dtype)


def _oracle_normalize_and_position(
    pooled: torch.Tensor, block_start_rope: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    values = pooled.float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    marker = block_start_rope[0].float().view(-1, 1, 1) / 4096.0
    return (normalized + marker).to(dtype)


def _pool(monkeypatch, device: torch.device, num_pages: int = 70) -> QSAKVCache:
    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )
    return QSAKVCache(
        num_kv_heads=2,
        num_layers=2,
        head_dim=16,
        num_pages=num_pages,
        page_size=64,
        dtype=torch.bfloat16,
        device=device,
        index_num_kv_heads=1,
        index_head_dim=INDEX_DIM,
        compress_ratio=RATIO,
        layer_ids=(LAYER_ID,),
    )


def _backend_and_context(monkeypatch, pool: QSAKVCache, max_tokens: int = 2112):
    stride = max_tokens
    page_table = torch.stack(
        (
            torch.arange(max_tokens, device=pool.device),
            torch.arange(stride, stride + max_tokens, device=pool.device),
        )
    ).long()
    assert int(page_table.max()) < pool.k_cache(LAYER_ID).numel() // (2 * 16)
    context = SimpleNamespace(kv_cache=pool, page_table=page_table)
    monkeypatch.setattr(qsa_module, "get_global_ctx", lambda: context)
    config = SimpleNamespace(
        qwen4_args=SimpleNamespace(
            indexer_compress_ratio=RATIO,
            indexer_budget=BUDGET,
        )
    )
    return QSAAttnBackend(config), context


def _raw_keys(length: int, request_id: int, device: torch.device) -> torch.Tensor:
    base = torch.arange(length * INDEX_DIM, dtype=torch.float32).view(length, 1, INDEX_DIM)
    values = base / 97.0 + 1.0 + request_id * 100.0
    return values.to(device=device, dtype=torch.bfloat16)


def _rope_positions(length: int, request_id: int, device: torch.device) -> torch.Tensor:
    logical = torch.arange(length, device=device, dtype=torch.int64)
    return torch.stack(
        (logical, logical + 1000 * request_id, logical + 2000 * request_id)
    )


def _request(start: int, end: int, table_idx: int):
    return SimpleNamespace(
        cached_len=start,
        device_len=end,
        extend_len=end - start,
        table_idx=table_idx,
    )


def _batch(reqs, rope_chunks):
    lengths = [req.extend_len for req in reqs]
    return SimpleNamespace(
        reqs=reqs,
        padded_reqs=reqs,
        rope_positions=torch.cat(rope_chunks, dim=1),
        input_ids=torch.empty(sum(lengths), dtype=torch.long),
    )


def _oracle_completed(
    full_keys: torch.Tensor, full_rope: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = full_keys.shape[0] // RATIO
    if not groups:
        return (
            full_keys.new_empty((0, 1, INDEX_DIM)),
            full_rope.new_empty((3, 0)),
        )
    members = full_keys[: groups * RATIO].view(groups, RATIO, 1, INDEX_DIM)
    pooled = members.float().mean(dim=1).to(full_keys.dtype)
    starts = torch.arange(0, groups * RATIO, RATIO, device=full_keys.device)
    rope = full_rope.index_select(1, starts)
    return _oracle_normalize_and_position(pooled, rope, full_keys.dtype), rope


def _oracle_selection(
    query: torch.Tensor, keys: torch.Tensor, position: int
) -> tuple[torch.Tensor, int]:
    complete = (position + 1) // RATIO
    width = min(BUDGET // RATIO, complete)
    if width:
        score = torch.relu(
            query.float() @ keys[:complete, 0].float().transpose(0, 1)
        ).sum(dim=0) / math.sqrt(query.shape[-1])
        groups = torch.topk(score, width, sorted=True).indices
        rows = (
            groups[:, None] * RATIO + torch.arange(RATIO, device=groups.device)
        ).flatten()
    else:
        rows = torch.empty(0, dtype=torch.long, device=query.device)
    tail = torch.arange(complete * RATIO, position + 1, device=query.device)
    rows = torch.cat((rows, tail))
    return rows, rows.numel()


def _assert_incremental_state(
    backend: QSAAttnBackend,
    context,
    full_keys: torch.Tensor,
    full_rope: torch.Tensor,
    end: int,
    table_idx: int,
) -> None:
    expected, _ = _oracle_completed(full_keys[:end], full_rope[:, :end])
    starts = torch.arange(0, expected.shape[0] * RATIO, RATIO, device=full_keys.device)
    physical = context.page_table[table_idx].index_select(0, starts)
    compressed_rows = torch.div(physical, RATIO, rounding_mode="floor")
    actual = backend.kvcache.compressed_k_cache(LAYER_ID).index_select(
        0, compressed_rows.long()
    )
    assert torch.equal(actual, expected)

    latest_start = max(0, end - RATIO)
    latest = torch.arange(latest_start, end, device=full_keys.device)
    assert torch.equal(
        backend.kvcache.pending_group(LAYER_ID, table_idx, latest),
        full_keys.index_select(0, latest),
    )
    assert torch.equal(
        backend.kvcache.pending_rope_group(LAYER_ID, table_idx, latest),
        full_rope.index_select(1, latest).transpose(0, 1),
    )

    query = torch.ones(1, 4, INDEX_DIM, device=full_keys.device)
    selected, counts = select_qsa_logical_rows(
        query,
        actual,
        torch.tensor([end - 1], device=full_keys.device),
        compress_ratio=RATIO,
        token_budget=BUDGET,
    )
    expected_rows, expected_count = _oracle_selection(query[0], actual, end - 1)
    assert int(counts[0]) == expected_count
    assert torch.equal(selected[0, :expected_count].long(), expected_rows.long())


def _run_pattern(monkeypatch, pattern: list[int], device: torch.device):
    total = sum(pattern)
    pool = _pool(monkeypatch, device)
    backend, context = _backend_and_context(monkeypatch, pool)
    indexer = _RecordingSyntheticIndexer()
    full_keys = _raw_keys(total, 0, device)
    full_rope = _rope_positions(total, 0, device)
    start = 0
    for length in pattern:
        end = start + length
        req = _request(start, end, 0)
        batch = _batch([req], [full_rope[:, start:end]])
        backend.prepare_metadata(batch)
        backend._compress_current_keys(
            indexer, full_keys[start:end], LAYER_ID, batch
        )
        _assert_incremental_state(
            backend, context, full_keys, full_rope, end, table_idx=0
        )
        start = end
    return backend, context, indexer, full_keys, full_rope


@pytest.mark.parametrize(
    "pattern",
    ([1, 3], [3, 1], [5, 2, 1], [1, 1, 1, 1, 1, 1, 1, 1]),
)
def test_qsa_incremental_compression_matches_full_history_cpu(monkeypatch, pattern):
    torch.manual_seed(SEED)
    backend, context, indexer, full_keys, full_rope = _run_pattern(
        monkeypatch, list(pattern), torch.device("cpu")
    )
    expected, expected_rope = _oracle_completed(full_keys, full_rope)
    assert indexer.calls
    # Across all calls, every emitted completed group uses its block-start RoPE.
    recorded_rope = torch.cat([positions for _, positions in indexer.calls], dim=1)
    assert torch.equal(recorded_rope, expected_rope.cpu())
    starts = torch.arange(0, expected.shape[0] * RATIO, RATIO)
    rows = torch.div(context.page_table[0].cpu().index_select(0, starts), RATIO, rounding_mode="floor")
    assert torch.equal(
        backend.kvcache.compressed_k_cache(LAYER_ID).cpu().index_select(0, rows),
        expected.cpu(),
    )


def test_qsa_incremental_crosses_first_sparse_boundary_cpu(monkeypatch):
    torch.manual_seed(SEED)
    pattern = [2047, 1, 1, 1, 1, 1]
    backend, context, _, full_keys, full_rope = _run_pattern(
        monkeypatch, pattern, torch.device("cpu")
    )
    assert full_keys.shape[0] == 2052
    expected, _ = _oracle_completed(full_keys, full_rope)
    assert expected.shape[0] == 513
    query = torch.ones(1, 4, INDEX_DIM)
    selected, counts = select_qsa_logical_rows(
        query,
        expected,
        torch.tensor([2051]),
        compress_ratio=RATIO,
        token_budget=BUDGET,
    )
    oracle_rows, oracle_count = _oracle_selection(query[0], expected, 2051)
    assert oracle_count == BUDGET
    assert int(counts[0]) == BUDGET
    assert torch.equal(selected[0, :BUDGET].long(), oracle_rows.long())
    # The least-scoring complete group is the only omitted group.
    omitted = set(range(2052)) - set(oracle_rows.tolist())
    assert omitted == set(range(4))
    assert backend.kvcache is context.kv_cache


def test_qsa_multi_request_cache_and_pending_state_are_isolated_cpu(monkeypatch):
    torch.manual_seed(SEED)
    pool = _pool(monkeypatch, torch.device("cpu"))
    backend, context = _backend_and_context(monkeypatch, pool)
    indexer = _RecordingSyntheticIndexer()
    lengths = (5, 7)
    keys = [_raw_keys(length, req_id, torch.device("cpu")) for req_id, length in enumerate(lengths)]
    rope = [_rope_positions(length, req_id, torch.device("cpu")) for req_id, length in enumerate(lengths)]
    reqs = [_request(0, length, req_id) for req_id, length in enumerate(lengths)]
    batch = _batch(reqs, rope)
    backend.prepare_metadata(batch)
    backend._compress_current_keys(indexer, torch.cat(keys), LAYER_ID, batch)

    for req_id, length in enumerate(lengths):
        _assert_incremental_state(
            backend, context, keys[req_id], rope[req_id], length, req_id
        )
    first_expected, _ = _oracle_completed(keys[0], rope[0])
    second_expected, _ = _oracle_completed(keys[1], rope[1])
    assert not torch.equal(first_expected[0], second_expected[0])

    index_q = torch.ones(sum(lengths), 4, INDEX_DIM)
    physical, counts = backend._select_physical_rows(index_q, LAYER_ID, batch)
    offset = 0
    for req_id, length in enumerate(lengths):
        for local_position in range(length):
            row = offset + local_position
            assert int(counts[row]) == local_position + 1
            actual = set(physical[row, : int(counts[row])].tolist())
            expected = set(
                context.page_table[req_id, : local_position + 1].tolist()
            )
            assert actual == expected
        offset += length


def test_qsa_request_slot_reset_invalidates_pending_state_cpu(monkeypatch):
    torch.manual_seed(SEED)
    backend, context, _, old_keys, old_rope = _run_pattern(
        monkeypatch, [5, 2, 1], torch.device("cpu")
    )
    new_keys = _raw_keys(3, 1, torch.device("cpu"))
    new_rope = _rope_positions(3, 1, torch.device("cpu"))
    req = _request(0, 3, 0)
    batch = _batch([req], [new_rope])
    backend.prepare_metadata(batch)
    backend._compress_current_keys(
        _RecordingSyntheticIndexer(), new_keys, LAYER_ID, batch
    )
    assert torch.equal(
        backend.kvcache.pending_group(LAYER_ID, 0, torch.arange(3)), new_keys
    )
    with pytest.raises(RuntimeError, match="pending-key state is missing"):
        backend.kvcache.pending_group(LAYER_ID, 0, torch.tensor([4]))
    selected, counts = select_qsa_logical_rows(
        torch.ones(1, 4, INDEX_DIM),
        backend.kvcache.compressed_k_cache(LAYER_ID)[:0],
        torch.tensor([2]),
        compress_ratio=RATIO,
        token_budget=BUDGET,
    )
    assert int(counts[0]) == 3
    assert torch.equal(selected[0, :3], torch.arange(3, dtype=torch.int32))
    assert old_keys.shape[0] == old_rope.shape[1] == 8
    assert backend.kvcache is context.kv_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qsa_incremental_cache_public_surface_matches_cpu_oracle_cuda(monkeypatch):
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    _run_pattern(
        monkeypatch,
        [1, 1, 1, 1, 1, 1, 1, 1],
        torch.device("cuda"),
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    print(
        "STAGE3_QSA_METRIC",
        json.dumps(
            {
                "case": "cuda_incremental_cache",
                "device": torch.cuda.get_device_name(),
                "driver_runtime": torch.version.cuda,
                "elapsed_seconds": elapsed,
                "peak_memory_bytes": peak,
            },
            sort_keys=True,
        ),
    )
    assert peak < 1 << 30
