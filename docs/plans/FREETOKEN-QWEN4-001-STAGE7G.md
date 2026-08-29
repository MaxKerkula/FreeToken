# FREETOKEN-QWEN4-001 / Stage 7G Storage Identity Remediation

## Definition of done

Separate Git blob, LFS OID/metadata ETag, Xet hash, and transport-body ETag throughout the Step 9B manifest, downloader, partial/resume state, and receipts; preserve the lifetime transfer ledger at 14,137 bytes; migrate the three existing source receipts without network body transfer; pass a body-disabled real restart rehearsal and the complete 215-file dry run; commit one clean remediation commit without changing any binary artifact/runtime contract.

## Dependencies and hard boundaries

- Parent executor commit: `6b62c5057ae13deb76bf92a80e8f925e90d55929`.
- Runtime remains byte-identical at `0307a6114c57b0efc61bc17688f3288fe0bf1dc7`.
- Source inventory fingerprint remains `8572d200e31b344faff0fdaf0dc72aa4726c1f062443d4109531b62ca63f66eb`.
- Lifetime transfer ledger remains `14,137 / 135,252,480,565` bytes.
- HEAD/API metadata only; any new source response-body byte is a stop condition.
- Q3, FTEXPERT1, active FTW, PR257 runtime, placement, conversion arithmetic, and target byte totals are out of scope.

## Execution plan

| Step | Status | Validation |
|---|---|---|
| Preserve and hash v1 manifest, receipts, transfer ledger, source files, and accepted worktrees | DONE | Exact SHA/length/state inventory |
| Freeze all nine metadata identities and representative BF16/PLE/EXPERT identities using metadata-only queries | DONE | Commit/size/Git/LFS/Xet semantic checks; zero body bytes |
| Implement manifest v2 identity fields and generation/migration | DONE | 9 metadata + 206 weights; totals/fingerprint unchanged |
| Implement executor v2 metadata, body ETag, partial, resume, and receipt semantics | DONE | Focused Git/LFS/Xet and transport tests |
| Revalidate three existing files and migrate receipts with predecessor hashes | DONE | Local full hashes passed; zero new body bytes; ledger unchanged |
| Run body-disabled real restart rehearsal and full 215-file dry run | DONE | Reached tokenizer body boundary; no Xet mismatch; no body GET |
| Run regressions, compileall, diff-check, source-delta audit, and adversarial review | DONE | 168 non-overlapping tests passed; compileall/diff-check passed; independent findings remediated |
| Create evidence, regenerate handoff, commit, and verify clean authorities | DONE | One commit with exact parent; runtime/history unchanged |

## Validation commands

- Focused and complete pytest through the accepted Stage 7F Python environment.
- `python -m compileall -q python/freetoken`
- `git diff --check`
- Stage 7G executor `--execute` without `--allow-network-body` against the actual restart state.
- Stage 7G executor `--dry-run` against all 215 manifest rows.
