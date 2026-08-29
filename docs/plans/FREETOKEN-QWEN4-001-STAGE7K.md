# FREETOKEN-QWEN4-001 Stage 7K plan

Task: `STAGE7K-STEP9B-ACTIVE-FTW-TERMINAL-PADDING-REMEDIATION`

Definition of done: preserve all acquired source files, Q3, and 48 expert sidecars; prove the active FTW discrepancy is exactly one terminal aligned page; make FTW v1 publication deterministically reach the frozen 4,804,403,200-byte extent without changing any tensor key, offset, length, dtype, or payload byte; recover B4; finalize B5; and pass isolated C6 with no network body transfer, full-model construction, inference, source deletion, runtime modification, or artifact v2.

Dependencies:

- Executor parent `3153f5e8f39a22aeb3d4283dba75336e988b1ba7`.
- Runtime `0307a6114c57b0efc61bc17688f3288fe0bf1dc7` remains clean and byte-identical.
- Acquisition manifest SHA-256 `8e4074cd1a8950bfb19ebdfdd4c5154b66db3ed538ba99fe41221d1be9361e74` remains unchanged.
- Lifetime transfer ledger remains `135252480565 / 135252480565`; all continuation work is body-disabled.

| Step | Status | Validation |
|---|---|---|
| Freeze and audit the current 4,804,399,104-byte FTW candidate | DONE | Index, shard, tensor inventory, hashes, and 4-KiB reconciliation |
| Implement the smallest FTW-v1 terminal-padding contract | DONE | Focused unit tests; no tensor metadata or payload changes |
| Prove recovery/adoption of the existing active candidate | PENDING | Before/after tensor inventory and prefix hashes; exact 4,804,403,200 bytes |
| Run relevant executor, FTW, artifact, active, and regression tests | DONE | `148 passed`; `compileall`; `git diff --check` |
| Commit the Stage 7K executor and regenerate the resume handoff | PENDING | Exact parent/commit, clean worktree, pinned zero-body command |
| Resume body-disabled B4 recovery, B5 finalization, and isolated C6 | PENDING | Receipts, known-total reconciliation, C6 static-only PASS |
| Close out machine, source-retention, ledger, cache, and evidence audits | PENDING | 215 retained sources, unchanged ledger/runtime/manifest, reserve checks |
