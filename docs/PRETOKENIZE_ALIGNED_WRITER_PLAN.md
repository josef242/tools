# Plan: pre_tokenize writes doc-aligned shards natively (rechunk sensibilities at birth)

**Goal (Josef, 2026-07-13/14):** stop generating river-chopped shards that need a
2-hour-per-corpus rechunk pass. New corpora come out of pre_tokenize already
doc-aligned, coprime, val-held-out, and manifest-verified — the same guarantees
`mara_fsdp2/tools/rechunk_doc_aligned.py` (as of cc86820) produces after the fact.

## Recon findings (2026-07-14)

- The river-chop is FOUR LINES: `ShardWriter.add()` (pre_tokenize.py:419-428)
  receives one whole tokenized doc per call and splits it across the fixed
  buffer. Everything upstream already speaks whole docs.
- BOS is already tokenizer-derived in the workers (`add_bos=True` via the
  tokenizer object) — the bos-audit-clean path. The writer can take `bos_id`
  from the same object; zero new literals.
- The atomic .tmp→rename discipline, stale-tmp cleanup, and resume-numbering
  (`_next_index`) all carry over unchanged.
- Docs arrive in FUTURE-COMPLETION order (parallel workers), i.e. approximately
  but not exactly input order. Consequence recorded below (val-holdout note).

## Design

### 1. Shared module: `common_fsdp2/doc_shard_writer.py`

```python
class DocShardWriter:
    def __init__(self, outdir, label, bos_id, cap=100_000_000, min_shard=5_000_000,
                 coprime=6, val_holdout=0, dtype=np.uint16): ...
    def add_doc(self, arr):   # ONE whole tokenized doc (starts with bos_id — asserted)
    def close(self):          # flush, coprime-finalize, write manifests; returns summary
```

Behavior:
- **Doc-aligned packing**: if the incoming doc doesn't fit the remaining buffer,
  flush first; a doc larger than cap is written directly as its own oversized
  shard (warned) — identical semantics to rechunk's planner.
- **Val routing**: while `val_tokens_written < val_holdout`, docs go to
  `<label>_val_NNNNNN.npy`; afterwards to `_train_`. Semantics note: with
  parallel tokenization this is "the first ~N tokens of docs in completion
  order" — approximately the corpus head, not exactly (rechunk's head-carve is
  exact). Acceptable: it is a val set, and slight order jitter is if anything
  a better sample. Document in --help.
- **Coprime finalize (YES — include it)**: at close(), if gcd(S_train, coprime)
  != 1, split the largest train shard at a BOS boundary (scan that ONE shard
  for BOS positions — one ~200MB read — reuse rechunk's find_split on a
  single-file river). Val excluded from the count, as in rechunk. Cost at
  generation time: seconds. This is the cheapest place coprimality will ever
  be enforceable.
- **Manifest**: same schema as rechunk (blake2b16 per shard, docs/tokens
  counts, coprime_ok, val_holdout_tokens) → `rechunk_doc_aligned.py
  --verify-only [--deep]` audits generated trees with NO changes. One
  verifier for every tree in the lab.
- **Crash-resume**: completed shards survive (atomic rename); the in-flight
  buffer's docs are lost exactly as today, but now the loss is whole-doc (no
  torn doc at a shard edge). Manifests are written only at close(); a
  resumed-and-completed run rebuilds counts by re-reading its own shards
  (header-only + BOS count pass) before writing manifests. If val files
  already satisfy the holdout at resume, route straight to train.
- **Bounded fds**: writer holds one output buffer; the finalize scan uses the
  open/close header pattern from rechunk cc86820.

### 2. pre_tokenize integration (small diff)

- Replace `ShardWriter` with `DocShardWriter` (constructed with
  `TOK.bos_id`, `--shard-size` as cap, new `--val-holdout` /
  `--val-holdout 0` to disable, `--coprime 6` default).
- `_flush_done` already hands whole docs to `add()` → rename to `add_doc()`.
- Import path: pre_tokenize (V:\code\tools repo) adds common_fsdp2 to
  sys.path the same way mara tools do (`../common_fsdp2` relative to repo
  root — confirm the relative geometry from V:\code\tools).
- Keep a `--legacy-river` escape hatch for one release in case some tree
  intentionally wants the old behavior (probably nobody; remove later).

### 3. rechunk refactor (phase 2, optional)

rechunk keeps its proven slice-based writer for now. Once DocShardWriter has
produced a real corpus end-to-end, rechunk's write path can be swapped to feed
docs into the same writer (river → docs → writer), deleting the duplicated
manifest/verify/coprime code. Not load-bearing; do when convenient.

### 4. Testing

- Port scratchpad test_rechunk_v2.py patterns: synthetic docs through the
  full pre_tokenize pipeline (multiprocessing on), assert: every shard starts
  with BOS, no doc torn (concat == sum of docs as a SET — completion order
  differs), coprime, val ~= holdout, manifests verify with the rechunk
  verifier, crash-resume (kill mid-run, resume, verify).
- One real small corpus (e.g. mid_gsm8k regen) as the acceptance run,
  verified with --verify-only --deep.

## Answered questions

- **Should it include coprime? YES** — at generation it costs one shard split
  at close (seconds); post-hoc it costs a rechunk. The orbit bug (W=8 x S=60)
  is exactly the class of defect that should be impossible by construction.
- Val default: 50_000_000 (see mara memory: supports millinat A/B floors;
  min(50M, ~0.5% of corpus) for small groups — writer warns if holdout > 2%
  of what it wrote).

## Effort

Writer module ~200 lines (half lifted from rechunk), pre_tokenize diff ~30
lines, tests ~150. Two repos touched (common_fsdp2, V:\code\tools). Roughly a
half-day careful build + the acceptance regen.

Status: PLAN ONLY (this doc). Implementation awaiting go-ahead.
