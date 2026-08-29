# FREETOKEN-QWEN4-001 Stage 7J — Q3 Acceleration

## Definition of done

Replace the day-scale scalar Q3 conversion loop with the smallest bounded batched implementation that is byte-identical to the frozen Q3_PLE_32 codec, materially uses the available CPU or RTX 5070, and can safely resume Step 9B without changing the artifact format, source manifest, runtime, or lifetime transfer ledger.

## Frozen dependencies

- Parent executor commit: `32217a407cbf2c829ecc01f2e26ea391660da278`
- PR257 runtime: `0307a6114c57b0efc61bc17688f3288fe0bf1dc7`
- Source inventory: `8572d200e31b344faff0fdaf0dc72aa4726c1f062443d4109531b62ca63f66eb`
- Lifetime transfer ledger: `51,257,444,615 / 135,252,480,565` bytes
- Binary contract: 128 segments, 70 bytes/row, 22,400,107,520 total Q3 bytes

## Plan

1. **DONE** — Benchmark byte-identical batched CPU and CUDA candidates on deterministic synthetic rows.
2. **DONE** — Select the smallest implementation meeting exact byte equality and bounded-memory requirements.
3. **DONE** — Implement the selected batched codec without changing scalar reference behavior or binary layout.
4. **DONE** — Add differential tests covering random rows, edge values, BF16 rounding boundaries, refinement convergence, chunk invariance, and 128-segment writer identity.
5. **DONE** — Benchmark sustained throughput and verify projected production duration and CPU/GPU/RAM bounds.
6. **DONE** — Run executor, Q3, identity-v2, Windows recovery, and manifest regressions plus `compileall` and `git diff --check`.
7. **DONE** — Commit the isolated executor change and regenerate the resume handoff. Resume the real Q3 component only under the new exact commit.

## Validation commands

- `python -m pytest -q tests/checkpoint/test_q3_ple_writer.py tests/checkpoint/test_step9b_executor.py tests/checkpoint/test_step9b_executor_contract.py tests/checkpoint/test_step9b_identity_v2.py tests/checkpoint/test_step9b_windows_atomic.py tests/checkpoint/test_step9b_ple_selector.py`
- `python -m compileall -q python/freetoken`
- `git diff --check`
- Synthetic scalar-versus-batched byte differential with deterministic random and adversarial BF16-scale rows
- Bounded sustained throughput benchmark on the selected backend

## Safety boundary

No real source download, no source deletion, no real Q3 restart, no expert acquisition, no runtime modification, and no inference occur until the accelerated implementation passes byte-exact review and receives a pinned local commit.
