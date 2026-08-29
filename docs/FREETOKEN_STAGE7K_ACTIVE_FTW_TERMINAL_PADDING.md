# Stage 7K — active FTW terminal-padding remediation

## Outcome

Stage 7K preserves the frozen active FTW v1 extent of `4,804,403,200` bytes without changing a tensor or introducing artifact v2. The production tensor stream naturally ends at `4,804,399,104` bytes. The exact `4,096`-byte difference is represented as one zero-filled terminal compatibility page.

## Root cause

The accepted analytical layout included the two-byte global PLE scale in the active resident-byte calculation. FTW alignment turns that scalar into one 4-KiB page. Production correctly binds the global PLE scale once in `ple-q3.json`, so the active tensor stream contains no duplicate PLE-scale tensor and ends one aligned page earlier.

The real candidate contains 1,698 tensor entries. Its final tensor ends exactly at byte `4,804,399,104`. Existing tensor payloads reconcile as:

```text
packed NVFP4 weights       1,772,748,800
FP8 block scales             221,593,600
FP16 row globals               4,162,176
protected tensors          2,804,402,200
raw tensor bytes           4,802,906,776
inter-entry alignment          1,492,328
tensor-stream extent       4,804,399,104
terminal compatibility page       4,096
frozen active extent       4,804,403,200
```

The page is not a tensor and does not represent a second PLE scale. FTW readers traverse indexed tensor entries and ignore the reserved tail.

## Recovery contract

`ensure_ftw_terminal_padding()` accepts only FTW v1 with 4-KiB alignment and only these two states:

- the exact unpadded production extent, followed by appending and fsyncing one zero page;
- the exact frozen extent with an already published or recoverable zero page.

It rejects nonterminal shard disagreement, nonzero tail bytes, partial pages, oversize tails, unknown extents, malformed geometry, and incompatible format/alignment. Publication order is shard append and fsync, then bounded Windows-safe atomic index replacement. A crash after append but before index replacement is recovered by validating and adopting the exact zero tail. Repeated recovery is idempotent.

Only the final shard `nbytes` and index `total_bytes` change. Tensor keys, order, offsets, lengths, dtypes, shapes, kinds, and all pre-tail bytes remain unchanged.

## Gate boundaries

- Source bodies remain retained and are not downloaded again.
- The lifetime transfer ledger remains `135,252,480,565 / 135,252,480,565`.
- Q3 and all 48 expert sidecars remain unchanged.
- PR257 runtime commit `0307a6114c57b0efc61bc17688f3288fe0bf1dc7` remains unchanged.
- B5 and isolated C6 may proceed only after B4 publishes a valid receipt at the frozen extent.
- No full model, inference, generation, serving, or Step 10 work is authorized.

