from __future__ import annotations

import json
import math
import time

import pytest
import torch

from freetoken.attention.qsa import select_qsa_logical_rows
from freetoken.kernel.triton.qsa import qsa_sparse_gqa


SEED = 38038
COMPRESS_RATIO = 4
TOKEN_BUDGET = 2048
OUTPUT_WIDTH = TOKEN_BUDGET + COMPRESS_RATIO - 1
QUERY_POSITIONS = (
    0,
    1,
    2,
    3,
    4,
    2047,
    2048,
    2049,
    2050,
    2051,
    2052,
    4095,
    8191,
    65535,
)


def _official_contiguous_oracle(
    index_q: torch.Tensor,
    compressed_keys: torch.Tensor,
    query_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent eager transcription of the pinned official indexer semantics.

    This oracle intentionally does not call any FreeToken selection or compaction
    helper. Its scope is the serving topology supported by FreeToken: one
    contiguous causal history per request.
    """
    rows = torch.full(
        (query_positions.numel(), OUTPUT_WIDTH), -1, dtype=torch.int32
    )
    counts = torch.empty(query_positions.numel(), dtype=torch.int32)
    offsets = torch.arange(COMPRESS_RATIO, dtype=torch.long)
    for query_row, position_tensor in enumerate(query_positions.cpu()):
        position = int(position_tensor)
        visible = position + 1
        complete_groups = visible // COMPRESS_RATIO
        width = min(TOKEN_BUDGET // COMPRESS_RATIO, complete_groups)
        if width:
            query = index_q[query_row].cpu().float()
            keys = compressed_keys[:complete_groups, 0].cpu().float()
            scores = torch.relu(query @ keys.transpose(0, 1)).sum(dim=0)
            scores /= math.sqrt(query.shape[-1])
            groups = torch.topk(scores, width, sorted=True).indices
            selected = (groups[:, None] * COMPRESS_RATIO + offsets).flatten()
        else:
            selected = torch.empty(0, dtype=torch.long)
        tail = torch.arange(
            complete_groups * COMPRESS_RATIO, visible, dtype=torch.long
        )
        selected = torch.cat((selected, tail))
        rows[query_row, : selected.numel()] = selected.to(torch.int32)
        counts[query_row] = selected.numel()
    return rows, counts


def _score_separated_fixture(
    positions: tuple[int, ...] = QUERY_POSITIONS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_positions = torch.tensor(positions, dtype=torch.int64)
    complete_groups = (max(positions) + 1) // COMPRESS_RATIO
    dim = 8
    index_q = torch.zeros(len(positions), 4, dim, dtype=torch.float32)
    index_q[:, :, 0] = torch.tensor([1.0, 1.5, 2.0, 2.5])
    compressed_keys = torch.zeros(complete_groups, 1, dim, dtype=torch.float32)
    # Strictly increasing positive scores make exact top-k order authoritative.
    compressed_keys[:, 0, 0] = torch.arange(
        1, complete_groups + 1, dtype=torch.float32
    )
    return index_q, compressed_keys, query_positions


def _live_rows(rows: torch.Tensor, counts: torch.Tensor, index: int) -> torch.Tensor:
    return rows[index, : int(counts[index])].long()


def _assert_group_and_tail_invariants(
    selected: torch.Tensor, position: int, expected_count: int
) -> None:
    assert selected.numel() == expected_count
    assert torch.all((selected >= 0) & (selected <= position))
    assert torch.unique(selected).numel() == selected.numel()
    visible = position + 1
    complete_groups = visible // COMPRESS_RATIO
    tail = torch.arange(complete_groups * COMPRESS_RATIO, visible)
    if tail.numel():
        assert torch.equal(selected[-tail.numel() :].cpu(), tail)
        selected = selected[: -tail.numel()]
    assert selected.numel() % COMPRESS_RATIO == 0
    if selected.numel():
        groups = selected.view(-1, COMPRESS_RATIO)
        assert torch.equal(
            groups,
            groups[:, :1] + torch.arange(COMPRESS_RATIO, device=groups.device),
        )
        assert torch.all(groups[:, 0] % COMPRESS_RATIO == 0)


def _independent_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rows: torch.Tensor,
    counts: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    output = torch.zeros_like(q, dtype=torch.float32)
    gqa = q.shape[1] // k.shape[1]
    for query_row in range(q.shape[0]):
        selected = rows[query_row, : int(counts[query_row])].long().cpu()
        for kv_head in range(k.shape[1]):
            heads = slice(kv_head * gqa, (kv_head + 1) * gqa)
            scores = torch.einsum(
                "hd,td->ht",
                q[query_row, heads].cpu().float(),
                k[selected, kv_head].cpu().float(),
            ) * scale
            probabilities = torch.softmax(scores, dim=-1)
            output[query_row, heads] = torch.einsum(
                "ht,td->hd", probabilities, v[selected, kv_head].cpu().float()
            )
    return output


def test_qsa_unique_score_selection_matches_pinned_official_oracle_cpu():
    torch.manual_seed(SEED)
    index_q, compressed_keys, positions = _score_separated_fixture()
    expected_rows, expected_counts = _official_contiguous_oracle(
        index_q, compressed_keys, positions
    )
    actual_rows, actual_counts = select_qsa_logical_rows(
        index_q,
        compressed_keys,
        positions,
        compress_ratio=COMPRESS_RATIO,
        token_budget=TOKEN_BUDGET,
    )
    assert torch.equal(actual_counts.cpu(), expected_counts)
    assert torch.equal(actual_rows.cpu(), expected_rows)

    for row, position in enumerate(QUERY_POSITIONS):
        selected = _live_rows(actual_rows, actual_counts, row)
        assert selected.numel() <= OUTPUT_WIDTH
        assert torch.all(selected <= position)
        if position <= 2050:
            assert set(selected.tolist()) == set(range(position + 1))

    boundary = QUERY_POSITIONS.index(2051)
    boundary_rows = _live_rows(actual_rows, actual_counts, boundary)
    assert actual_counts[boundary].item() == TOKEN_BUDGET
    assert set(boundary_rows.tolist()) == set(range(4, 2052))
    assert not any(token in boundary_rows.tolist() for token in range(4))

    after = QUERY_POSITIONS.index(2052)
    assert actual_counts[after].item() == TOKEN_BUDGET + 1
    assert _live_rows(actual_rows, actual_counts, after)[-1].item() == 2052


@pytest.mark.parametrize("key_value", [0.0, -1.0])
def test_qsa_tied_scores_obey_order_insensitive_official_invariants_cpu(key_value):
    positions = torch.tensor([2051, 2052, 4095], dtype=torch.int64)
    index_q = torch.ones(positions.numel(), 4, 8)
    compressed_keys = torch.full((1024, 1, 8), key_value)
    rows, counts = select_qsa_logical_rows(
        index_q,
        compressed_keys,
        positions,
        compress_ratio=COMPRESS_RATIO,
        token_budget=TOKEN_BUDGET,
    )
    for row, position in enumerate(positions.tolist()):
        complete = (position + 1) // COMPRESS_RATIO
        tail = (position + 1) % COMPRESS_RATIO
        expected_count = min(complete, TOKEN_BUDGET // COMPRESS_RATIO) * 4 + tail
        _assert_group_and_tail_invariants(
            _live_rows(rows, counts, row), position, expected_count
        )


def test_qsa_float32_sparse_attention_matches_independent_eager_oracle_cpu():
    torch.manual_seed(SEED)
    positions_tuple = (2051, 2052, 4095)
    index_q, compressed_keys, positions = _score_separated_fixture(positions_tuple)
    selected, counts = select_qsa_logical_rows(
        index_q,
        compressed_keys,
        positions,
        compress_ratio=COMPRESS_RATIO,
        token_budget=TOKEN_BUDGET,
    )
    q = torch.randn(len(positions_tuple), 4, 16, dtype=torch.float32)
    k = torch.randn(max(positions_tuple) + 1, 2, 16, dtype=torch.float32)
    v = torch.randn_like(k)
    scale = 16**-0.5
    actual = qsa_sparse_gqa(q, k, v, selected, counts, scale)
    expected = _independent_sparse_attention(q, k, v, selected, counts, scale)
    error = (actual.float() - expected).abs()
    print(
        "STAGE3_QSA_METRIC",
        json.dumps(
            {
                "device": "cpu",
                "dtype": "float32",
                "max_abs_error": error.max().item(),
                "mean_abs_error": error.mean().item(),
            },
            sort_keys=True,
        ),
    )
    torch.testing.assert_close(actual.float(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qsa_cuda_selection_and_compaction_match_independent_oracle():
    torch.manual_seed(SEED)
    index_q, compressed_keys, positions = _score_separated_fixture()
    expected_rows, expected_counts = _official_contiguous_oracle(
        index_q, compressed_keys, positions
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    actual_rows, actual_counts = select_qsa_logical_rows(
        index_q.cuda(),
        compressed_keys.cuda(),
        positions.cuda(),
        compress_ratio=COMPRESS_RATIO,
        token_budget=TOKEN_BUDGET,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert torch.equal(actual_counts.cpu(), expected_counts)
    assert torch.equal(actual_rows.cpu(), expected_rows)
    peak = torch.cuda.max_memory_allocated()
    print(
        "STAGE3_QSA_METRIC",
        json.dumps(
            {
                "case": "cuda_selection_compaction",
                "device": torch.cuda.get_device_name(),
                "driver_runtime": torch.version.cuda,
                "elapsed_seconds": elapsed,
                "peak_memory_bytes": peak,
            },
            sort_keys=True,
        ),
    )
    assert peak < 1 << 30


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qsa_cuda_bf16_sparse_gqa_matches_independent_eager_oracle():
    torch.manual_seed(SEED)
    positions_tuple = (2051, 65535)
    index_q, compressed_keys, positions = _score_separated_fixture(positions_tuple)
    selected, counts = _official_contiguous_oracle(index_q, compressed_keys, positions)
    dim = 64
    q_cpu = torch.randn(len(positions_tuple), 8, dim, dtype=torch.bfloat16)
    k_cpu = torch.randn(max(positions_tuple) + 1, 2, dim, dtype=torch.bfloat16)
    v_cpu = torch.randn_like(k_cpu)
    scale = dim**-0.5
    expected = _independent_sparse_attention(
        q_cpu, k_cpu, v_cpu, selected, counts, scale
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    actual = qsa_sparse_gqa(
        q_cpu.cuda(),
        k_cpu.cuda(),
        v_cpu.cuda(),
        selected.cuda(),
        counts.cuda(),
        scale,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    actual_cpu = actual.cpu().float()
    error = (actual_cpu - expected).abs()
    peak = torch.cuda.max_memory_allocated()
    print(
        "STAGE3_QSA_METRIC",
        json.dumps(
            {
                "case": "cuda_bf16_sparse_gqa",
                "device": torch.cuda.get_device_name(),
                "driver_runtime": torch.version.cuda,
                "elapsed_seconds": elapsed,
                "peak_memory_bytes": peak,
                "max_abs_error": error.max().item(),
                "mean_abs_error": error.mean().item(),
            },
            sort_keys=True,
        ),
    )
    assert peak < 1 << 30
    torch.testing.assert_close(actual_cpu, expected, rtol=2e-2, atol=2e-2)
