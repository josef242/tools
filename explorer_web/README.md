# Dataset Explorer Web

A FastAPI + vanilla-JS web frontend for `dataset_explorer.py`. No build step.

## Run

```bash
# needs the data stack (numpy/pandas/pyarrow) + fastapi/uvicorn
# on this machine that's the bookclean conda env:
~/miniconda3/envs/bookclean/bin/python explorer_web/server.py --port 8765
# bind 0.0.0.0 to reach it from another machine on the LAN
```

Then open http://localhost:8765.

## What it does

- **Library (managed datasets)**: a persistent registry of known datasets —
  name, tags, notes, kind (text/tokenized), lineage (`derived_from` + the job
  that produced it), cached vitals (records / tokens from pre_tokenize
  manifests / size), and remembered open options (text field, tokenizer) so a
  registered dataset opens correctly with one click. Registry lives in a JSON
  file (`--registry`, default `~/whynot/traindata/registry.json`) — point every
  rig at the same NAS path to share one library. Register/edit/unregister are
  metadata-only: data files are never touched. "Scan dir" lists registerable
  candidates in one directory (non-recursive).

- **Load datasets at runtime** (no argv): any path the CLI accepts — .jsonl,
  .parquet, .jsonl.zst, .npy shard dirs (with tokenizer options / raw-shard
  mode in the load form). Several datasets can be open at once.
- **Browse**: page through the whole dataset or any result set, click a record
  for the full view. `goto #` jumps to an offset.
- **Search**: `findall`-style text/regex/multi-term AND search, or metadata
  queries via the metaindex. Results are stored as named result sets. A search
  can be scoped **within an existing set** (intersection) — this is the
  "filter a set from a set" flow, which the CLI doesn't have.
- **Sets**: same persistent result sets as the CLI (`.dataset_explorer_cache/
  *.sets.gz`, shared both ways). Adds union/intersect/subtract combinators and
  **export to JSONL** with include/exclude sets.
- **Export**: recipe (include/exclude sets) → live plan with exact byte sizes →
  single JSONL / shard-mirror / split-to-size, atomic writes, auto-registered
  with lineage.
- **Filters** (own tab — the second library-level tab beside Library):
  corpus-agnostic named rule definitions (contains / startswith / len_lt /
  len_gt / regex / python expr) stored in the registry, versioned on edit.
  Evaluate against any open dataset: **shallow** (evenly-spaced sample →
  per-10k rate table) or **materialize** (full scan → per-rule sets
  `<filter>.<rule>` plus `<filter>.any`, so drops are browsable before they're
  destructive). Every evaluation is recorded to the registry (keyed by
  filter+corpus+version+sample; re-runs replace their column, capped at 24)
  and rendered in the **Evaluations panel: one column per run, corpora side
  by side** — the corpus A/B comparison is one screen. History is
  NAS-shared and survives restarts; "clear history" per filter.
- **Transforms** (own tab — the fourth noun): corpus-agnostic **ordered
  rewrite chains** (pattern → replacement, per field; each row picks a
  pattern mode — **regex**, **literal** (exact substring, paste-safe), or
  **glob** (exact except `*` = shortest within-line stretch) — plus a
  **line** checkbox that makes the match take its entire line, newline
  included), versioned
  independently of filters. Where a filter is a predicate composing as set
  algebra, a transform is a function composing by sequence — order matters,
  and optional **fixpoint mode** re-runs the chain per record until nothing
  changes (cap 10; preview flags non-convergence loudly). Preview dry-runs
  the chain on a sample — optionally on a chosen filter's *survivors* — with
  per-scrub hit rates, chars removed, and before/after diffs; runs are
  recorded per corpus in a side-by-side history. Transforms apply at export
  passes only (source data never mutated): pick one on the **Tokenize** tab
  (stream mode applies it in-stream via the view manifest; file mode rewrites
  the intermediate — token-identical by test) or on the **Export** tab (all
  three modes rewrite while writing). Exact per-scrub counts land in job
  results and the derived artifact's recipe, which records
  `(filter@version, transform@version)`. Exact per-scrub counts land in the job result
    and the tokenized child's recipe — ledger-ready. Materialized drop sets
    are stamped with a rules-only hash, so editing scrubs never invalidates
    them.
- **Tokenize**: full pre_tokenize.py orchestration — extraction preview on real
  records, preflight (tokenizer/vocab/dtype, globs, template, resume detection),
  subprocess job with parsed progress, result auto-registered with the full
  recipe (clonable; "Tokenized versions" table per source).
  - **Composition**: pick include/exclude sets and/or a filter on the Tokenize
    tab to tokenize a cleaned *view* of the dataset. Filter sets are
    materialized first if their current-version sets don't already exist; a
    live plan line shows kept/dropped records and byte size. Two intermediate
    modes (the materialization-tier choice):
    - **stream (default, no copy)**: writes a tiny view manifest
      (`<label>.view.json` + `.npz` skip-ordinal arrays) into the output dir
      and pre_tokenize reads the SOURCE through it (`--view-manifest`),
      dropping excluded records in-stream. Zero export cost — hours saved on
      big corpora. Guards: per-file record counts verified at EOF (source
      drift is fatal), and a composition hash stamp refuses resuming the
      output dir with a different composition.
    - **materialize JSONL**: export a plain `*_cleaned` intermediate
      (registered with lineage; browsable) and tokenize that — for when the
      cleaned corpus is itself a wanted artifact. Uncheck "keep" for an
      ephemeral intermediate (deleted after success; the tokenized child then
      descends directly from the source).
    Either way the tokenized entry's recipe records the whole composition
    (sets, filter+version, mode, counts) — stream and materialize are
    token-identical by construction (and by test).
- **Dedup**: run neardupe (GPU or CPU), inspect clusters, jump to member
  records, dry-run / commit prunes.
- **Jobs**: every slow operation is a job on a per-dataset worker thread;
  logs (the explorer's normal print output) stream live over SSE.

## Moved datasets (automatic cache adoption)

Caches are keyed on the path spelling AS GIVEN (`absolute()`, never resolved
through symlinks/mapped drives). Directory datasets carry a
`.dataset_explorer_cache/pathkey.json` marker recording that spelling. On
every load the server checks for caches keyed to a different spelling (marker
mismatch, or foreign hash tags when unmarked) and adopts them automatically —
rename-only, idempotent, content re-validated. If adoption finds MULTIPLE
candidate generations for one artifact, the load stops before deriving
anything and the UI shows a keep/discard chooser (dates + sizes); losers are
sidelined as `*.superseded`, never deleted, then the load retries itself.
CLI equivalent: `dataset_explorer.py <path> --migrate-cache` (warns on open,
never auto-adopts, so scripts are never surprised).

## Architecture notes

- One `DatasetExplorer` per loaded dataset, owned by **one worker thread**
  that serializes all mutating/slow operations — the class is not thread-safe
  and this preserves its REPL-era assumptions. Fast reads (record fetch via
  the line index) bypass the queue so browsing stays live during long jobs.
- stdout is swapped for a thread-local proxy: worker threads write into their
  current job's log, everything else falls through to the real stdout. Rich
  and tqdm are disabled in-process so logs are plain text.
- `dataset_explorer.py` gained one kwarg (`non_interactive=True`) which
  auto-selects "build complete index" at the large-JSONL prompt, and a
  module-level `PROGRESS_HOOK` — a structured progress channel
  (`report_progress(stage, done, total, main)`) instrumenting every long
  loop: indexing, decompress, npy decode, search scans, metaindex build,
  neardupe sketch/match, prune, export. The server installs the hook and
  computes %/ETA per stage; jobs stream `progress` SSE events rendered as
  `stage [████░░] 42.0% (n/total) eta 3m10s` (two levels: file i/N + within-
  file). The CLI leaves the hook as None — every call is a no-op there.
- The log capture understands `\r` progress rewrites (a `print(..., end='\r')`
  loop shows as one updating log line, not nothing).
- All disk caches (decompression, npy decode, line index, metaindex, neardupe
  signatures, result sets) are the CLI's own — the two frontends share them.
