# FREETOKEN-QWEN4-001 Stage 7I plan

Task: `STAGE7I-STEP9B-PLE-LAYER-SELECTOR-REMEDIATION`

## Definition of done

- Preserve executor commit `d391b4a6e6a31ffa206f7c8339920e2130e174f3`, runtime commit `0307a6114c57b0efc61bc17688f3288fe0bf1dc7`, acquisition manifest v2, all 19 acquired source files, and lifetime transfer ledger `51,257,444,615 / 135,252,480,565`.
- Make the Step 9B executor validate and select the frozen production PLE source layer represented by the immutable local source index instead of requesting nonexistent layer 2.
- Fail closed unless the index contains exactly `shard_0..shard_127` plus the global `weight_scale` under the expected production layer.
- Prove the correction with synthetic/index-only tests and a body-disabled real-state rehearsal. Do not construct the real Q3 artifact.
- Create one local executor commit and regenerate the resume handoff. Do not resume network body acquisition or B3.

## Dependencies

- Parent executor commit: `d391b4a6e6a31ffa206f7c8339920e2130e174f3`.
- Immutable source revision: `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
- Acquisition manifest SHA-256: `8e4074cd1a8950bfb19ebdfdd4c5154b66db3ed538ba99fe41221d1be9361e74`.
- Existing local B1/B2 source and receipts; no new response-body transfer.

## Steps

1. `DONE` — Verify authorities, manifest, ledger, and acquired source state; create isolated Stage 7I worktree.
2. `DONE` — Implemented the smallest explicit PLE selector validation and focused regressions.
3. `DONE` — Ran focused executor/Q3 tests, relevant regression suite, compileall, and diff-check.
4. `DONE` — Completed independent read-only review and index/header-only real-state rehearsal.
5. `DONE` — Prepared evidence and the local commit for the regenerated resume handoff and clean closeout.

## Validation commands

```text
python -m pytest -q tests/checkpoint/test_step9b_ple_selector.py
python -m pytest -q tests/checkpoint/test_step9b_executor.py tests/checkpoint/test_step9b_executor_contract.py tests/checkpoint/test_step9b_identity_v2.py tests/checkpoint/test_step9b_windows_atomic.py tests/checkpoint/test_q3_ple_writer.py
python -m compileall -q python/freetoken
git diff --check
```

The real-state rehearsal must omit `--allow-network-body` and must stop before Q3 construction.
