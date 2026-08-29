# FreeToken Qwen4 Stage 7F Step 9B executor

Gate: `FREETOKEN-QWEN4-001 / STAGE7F-STEP9B-EXECUTOR-IMPLEMENTATION`

This branch adds the missing execution controller around the accepted Stage 7E
component writers. It does not change Q3, FTEXPERT1, active FTW v1, or the
pinned PR257 runtime. Stage 7F itself is a zero-real-payload gate.

## Safety model

`Step9BExecutor` is dry-run by default. A network response body is unreachable
unless both execution mode and explicit body permission are enabled. The CLI
therefore requires `--execute --allow-network-body`; `--execute` alone fails
before acquisition. The frozen manifest is the only source-file authority.

The downloader:

- resolves the immutable Hugging Face commit and checks file size, ETag, LFS
  OID, Xet identity, and bounded Safetensors-header identity before transfer;
- streams directly to `<filename>.partial` on Z:, with no Hugging Face cache;
- binds every partial to a durable identity sidecar and resumes only with an
  exact `Range` plus `If-Range` contract;
- rejects an ignored range, malformed `Content-Range`, missing or changed body
  ETag, oversized or undersized body, hash mismatch, and header mismatch;
- persists actual response-body bytes before accepting each chunk, including
  bytes received in a failed or over-cap attempt;
- enforces no more than two active response bodies and provides cancellation;
- promotes a source atomically only after all identities validate, then writes
  an atomic source receipt.

The controller rejects source payload without matching durable transfer-budget
provenance. This prevents an orphan partial or final source from bypassing the
upstream byte cap after a restart.

## Stage controller

The controller exposes explicit boundaries for:

1. B1: nine metadata/config/tokenizer files and a bound receipt.
2. B2: ten PLE source files, the accepted 128-segment Q3 writer, reader reopen,
   exact 22,400,107,520-byte validation, precommit, and final receipt.
3. B3: 48 ordered expert transactions, four source files per layer, accepted
   FTEXPERT1 writer, exact 1,419,776,000-byte extent, FileExpertSource reopen,
   and an independent layer receipt.
4. B4: four BF16 source files, accepted active conversion, exact
   4,804,403,200-byte FTW v1 contract, and receipt.
5. B5: final modular-manifest validation and exact 95,353,758,720-byte known
   component reconciliation.
6. C6: a process-isolated static reopen using only the pinned PR257 worktree.

Existing valid targets are never trusted by name or receipt alone. The
controller rehashes and reopens them, then can recover a missing final receipt.
Accepted component writers own atomic target promotion; the controller adds a
validated precommit document followed by an atomic final receipt. Incomplete
component partials are never promoted by the controller.

Every component receipt binds the builder commit, runtime commit where
relevant, source revision, source-inventory fingerprint, receipt-backed source
file hashes, target length/hash, format, and validation results.

## Capacity and environment gates

The disk gate is restart-aware. It computes:

`remaining verified source + remaining target + common conversion allowance + 64 GiB reserve`

rather than requiring the original peak-free threshold after every completed
file. With an empty source and target, the formula exactly reconciles to
309,257,827,893 bytes. Physical host availability is measured independently
and must remain at least 6,442,450,944 bytes; pagefile use is recorded but never
counted as model capacity.

The executor verifies the isolated `_pinned_tensor` extension hash and, in real
execution preflight, runs a bounded HostBank pin/device-alias probe. Triton,
TVM-FFI, Torch, compiler, and temporary caches are forced to the supplied Z:
toolchain root.

Source retirement is explicitly rejected by this controller. The real handoff
uses `source_retirement_authorized=false` and retains every verified source.

## C6 isolation

C6 launches a fresh child process with `PYTHONPATH` beginning at exactly the
pinned PR257 runtime source. It performs static-only checks of:

- hardware-fit markers and active FTW verification;
- Q3PLEFileTable construction;
- all 48 expert metadata entries and the 12/36 placement policy;
- QD4 FileExpertSource construction for each file-tier layer;
- graph-disabled and prefill-overlap-disabled policy, including forced-graph
  rejection;
- a tiny resident HostBank pin/device-pointer probe.

C6 does not instantiate the complete model, resident expert banks, production
GPU cache, KV cache, a model layer, a forward pass, generation, or a server.

## Validation

The Stage 7F suite uses a local deterministic HTTP server for clean transfer,
interruption, exact resume, ETag drift, ignored and malformed ranges,
over/undersized bodies, hashes, Safetensors headers, cancellation, concurrency,
failure isolation, receipt recovery, and byte-cap behavior. A hard kill-switch
test proves dry run cannot issue a real Hugging Face body GET. The full frozen
manifest test plans 9 metadata files, 206 weight files, 48 expert boundaries,
135,195,303,851 weight bytes, and the complete B1-B5/C6 order.

The production command is intentionally emitted only in the separately
regenerated Step 9 handoff. Stage 7F does not execute it.
