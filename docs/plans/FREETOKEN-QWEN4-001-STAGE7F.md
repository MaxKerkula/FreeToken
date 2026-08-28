# FREETOKEN-QWEN4-001 Stage 7F execution plan

Task: `STAGE7F-STEP9B-EXECUTOR-IMPLEMENTATION`

Definition of done: a dry-run-default, restartable Step 9B controller coordinates immutable source acquisition, B1-B5 conversion, receipts, stop gates, retained-source policy, and isolated C6 validation without changing frozen component formats or the pinned PR257 runtime. Synthetic transport/controller tests and the real 206-row manifest dry run pass with zero upstream model payload bytes, then one local commit and a non-executed Step 9 handoff are produced.

Dependencies: accepted builder `b64a342ea8e5ccac39a7619747b4a7b3e37466f3`; runtime `0307a6114c57b0efc61bc17688f3288fe0bf1dc7`; source manifest revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.

Validation: focused downloader, receipt/recovery, staged-controller, C6 subprocess, and accepted component-writer tests; full real-manifest dry run; `python -m compileall -q python/freetoken`; `git diff --check`; commit parent and clean-worktree checks.

| Step | Status | Work |
|---|---|---|
| F0 | DONE | Verify authorities, PR257 head, zero-payload state, and create isolated branch. |
| F1 | DONE | Map accepted writer/manifest/runtime seams and freeze executor contract. |
| F2 | DONE | Implement immutable transport, downloader, partial identity, byte budget, receipts, and logging. |
| F3 | DONE | Implement B1-B5 state controller, disk/RAM gates, retained-source policy, and CLI. |
| F4 | DONE | Implement isolated C6 static-validation subprocess controller and environment check. |
| F5 | DONE | Add synthetic HTTP, range/cap/cancellation, receipt crash-recovery, and controller tests. |
| F6 | DONE | Run synthetic end-to-end and real 206-row no-body dry run; capture required evidence. |
| F7 | DONE | Run accepted regressions, compileall, diff checks, and adversarial final review. |
| F8 | DONE | Commit the executor branch and regenerate the non-executed Step 9 handoff. |
| F9 | DONE | Verify clean worktrees, zero payload, evidence completeness, and final gate decision. |
