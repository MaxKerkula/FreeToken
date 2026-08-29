# FREETOKEN-QWEN4-001 Stage 7H execution plan

Task: `STAGE7H-STEP9B-WINDOWS-ATOMIC-SIDECAR-RECOVERY`

Definition of done: the accepted Stage 7G executor is extended only with bounded Windows atomic-JSON replacement retries and fail-closed orphan checkpoint recovery; the existing first-PLE partial is preserved byte-for-byte and its exact orphan identity is adopted without any network response-body bytes; the lifetime ledger, manifest, runtime, and B1 state remain unchanged; the body-disabled real restart derives the exact future range plan; all focused regressions pass; and a clean local Stage 7H commit plus pinned resume handoff and evidence are produced.

Dependencies:

- Historical Stage 7G executor at `313c043861df5c57dfd7c2f98ec168dc7a631d28` remains clean.
- Manifest v2 SHA-256 remains `8e4074cd1a8950bfb19ebdfdd4c5154b66db3ed538ba99fe41221d1be9361e74`.
- PR257 runtime remains `0307a6114c57b0efc61bc17688f3288fe0bf1dc7` and clean.
- Existing B1, partial body, sidecars, and transfer ledger remain available and unmodified until preservation evidence is captured.
- No body GET, Range request, Q3 conversion, or subsequent source acquisition is permitted.

Validation commands:

- Focused Stage 7F/7G/7H executor tests under `tests/checkpoint`.
- Z:-backed real Windows lock integration and repeated-publication stress tests.
- Body-disabled execution against the actual retained source state.
- `python -m compileall -q python/freetoken`.
- `git diff --check`.
- Post-recovery hashes/sizes for the partial, ledger, manifest, B1 receipt, runtime authority, and recovered sidecar.

Steps:

1. `DONE` — Freeze authorities and preserve pre-recovery evidence, including a full partial-body recovery fingerprint.
2. `DONE` — Audit executor-owned handles and map every orphan-adoption invariant.
3. `DONE` — Implement bounded Windows replace retry and fail-closed orphan recovery.
4. `DONE` — Run synthetic retry, Windows lock, stress, recovery, identity/resume, receipt, and transfer-budget regressions.
5. `IN PROGRESS` — Adopt the independently validated real orphan and run the body-disabled actual-state restart rehearsal.
6. `PENDING` — Produce Stage 7H evidence, commit locally, regenerate the Step 9B resume handoff, and verify clean closeout.
