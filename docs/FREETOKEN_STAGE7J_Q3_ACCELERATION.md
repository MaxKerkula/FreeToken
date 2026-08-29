# Stage 7J — Byte-Exact Q3 Acceleration

Stage 7J replaces only the production Q3 encoder's scalar row loop.  The
Q3_PLE_32 format, 128 logical segment boundaries, per-segment hashes, global
weight scale, manifest schema, source inventory, and PR257 runtime remain
unchanged.

## Reason for remediation

The first real conversion measured approximately 198 kB/s and projected about
31 hours for the 22,400,107,520-byte Q3 extent while using roughly one CPU
core.  The run was stopped before its first component receipt.  Its incomplete
target partial is not a valid component; all ten verified PLE source files and
the lifetime transfer ledger remain unchanged.

## Selected implementation

The production Safetensors path now reads bounded 131,072-row chunks and
executes the existing two-pass codec as batched Torch operations.  CUDA is
selected when available and the same batched arithmetic has a CPU fallback.
The scalar `quantize_block()` and `quantize_row()` functions remain the
reference authority.

Exactness protections include:

- the historical FP8-to-FP32 source conversion boundary;
- float64 scale and refinement arithmetic;
- sequential `cumsum` reductions matching the scalar left-fold order;
- per-block early-convergence state;
- the existing integer round-to-nearest-even BF16 scale conversion;
- final requantization against the stored BF16 scale;
- unchanged low-bit-first 3-bit packing;
- ordered writes and unchanged logical segment/hash construction.

## Evidence

- 50,000 deterministic FP8-origin rows: scalar and CUDA output byte-identical;
- SHA-256 of the 3,500,000-byte differential output:
  `dfe54d8a9d122d356cb46b21a7fda6e2daea7a91b404bcd4a808a9ab9e012f39`;
- measured differential speedup: 125.9x;
- 1,048,576-row sustained synthetic stream: 425,479 rows/s including source
  generation, host/device transfer, output materialization, and hashing;
- sustained projected encoding time: 12.53 minutes;
- 131,072-row warmed batch: 1,320,719 rows/s, projected 4.04 minutes for codec
  execution alone;
- peak measured GPU allocation: 1,329,070,080 bytes;
- focused executor/Q3/resume regression matrix: 107 passed;
- Q3 writer suite: 20 passed.

The real Q3 conversion is not restarted until this isolated branch is committed
and the resume handoff pins that exact commit.
