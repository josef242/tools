# pre_tokenize.py
> Unified pre-tokeniser that turns JSON/JSONL/Parquet datasets into fixed-size NumPy token shards for LLM training.

## What it does
Reads raw text datasets (JSON arrays, JSONL, zstd-compressed JSONL, Parquet, or
Scriptorium scanned-book formats), tokenises every document with a pluggable
tokenizer, and writes fixed-length `uint16`/`uint32` `.npy` token shards plus a
shared `token_cache.json`. Reach for it when you need to convert a corpus into
the on-disk shard format the training loop consumes. It is resumable (a per-label
`.manifest` records processed files + token counts), parallel (a process pool of
tokeniser workers), and supports optional English-only language filtering. This
is a standalone data-prep tool in `tools/`, not part of the optimizer/WD/Newton-Schulz
investigation families; it sits upstream of training, feeding shards + the
`token_cache.json` that the data loader reads.

## Usage
```bash
# JSONL files, text in the "text" field, default tiktoken tokenizer
python pre_tokenize.py data_jsonl tokens --field text

# HuggingFace tokenizer, Parquet input, 100M-token shards, custom field
python pre_tokenize.py parquet_dir tokens --input-format parquet \
       --tokenizer hf --tokenizer_path ./superbpe \
       --field body --shard-size 100000000

# Scriptorium scanned-book JSONL (one joined doc per book, per-page quality filter)
python pre_tokenize.py books_dir tokens --input-format scanned-book-jsonl --label books

# Process a frozen Scriptorium batch manifest (input_dir = the batch JSONL file)
python pre_tokenize.py batch_manifest.jsonl tokens --input-format batch --label tier1
```

The first positional arg is the input dir (or single file, or — for `--input-format
batch` — a batch JSONL file path). The second is the output dir where shards,
the `.manifest`, and the parent `token_cache.json` are written.

## Key arguments
| flag | default | meaning |
| --- | --- | --- |
| `input_dir` (positional) | — | Dir or single file with the dataset; a batch JSONL file when `--input-format batch` |
| `output_dir` (positional) | — | Where `.npy` shards, the `.manifest`, and `token_cache.json` go |
| `--input-format` | `auto` | `auto`/`json`/`jsonl`/`parquet`/`scanned-book-jsonl`/`batch` |
| `--field` | `text` | JSON key / Parquet column holding the text (ignored if `--format` given) |
| `--format` | — | Full Python format string (NestedFormatter); overrides `--field` |
| `--tokenizer` | `tiktoken` | Tokenizer backend: `llama`/`hf`/`tiktoken`/`claude` |
| `--tokenizer_path` | — | Path/ID for HF tokenizers (also used as cache tokenizer name when `--tokenizer hf`) |
| `--shard-size` | `1e8` (100M) | Tokens per `.npy` shard |
| `--dtype` | `auto` | `uint16`/`uint32`/`auto` (auto = uint16 if vocab < 65536 else uint32) |
| `--label` | `data` | Filename prefix; also the manifest name and `token_cache` group key |
| `--workers` | `cpu_count()//2` | Tokenisation worker processes |
| `--batch-size` | `1000` | Rows per Parquet read batch |
| `--filter-english` | off | Keep only English docs (needs `lang_detect` / a detection lib) |
| `--lang-threshold` | `0.8` | Confidence threshold for language detection |
| `--lang-backend` | `auto` | `auto`/`fasttext`/`langdetect`/`langid` |
| `--lang-sample-size` | `500` | Max chars sampled per doc for language detection |
| `--view-manifest` | — | Tokenize a COMPOSED view of the source (dataset-explorer `<label>.view.json`): the manifest's file list is used verbatim (no glob) and per-file record ordinals in its skip arrays are dropped in-stream — no intermediate copy. Incompatible with `scanned-book-jsonl`/`batch`. |

Scanned-book page-filter thresholds (only used for `scanned-book-jsonl`/`batch`):
`--max-non-dict-ratio` (0.50), `--min-alpha-ratio` (0.60), `--min-char-count`
(150), `--max-repetition-ratio` (0.05), and `--include-matter` (keep front/back
matter pages, default drops them).

## Notes
- **Resume / manifest**: writes `<output_dir>/<label>.manifest` (tab-separated
  `path\ttoken_count`). Re-running skips already-processed files and accumulates
  token totals; an old format without token counts is also read.
- **View manifests** (`--view-manifest`): the dataset-explorer web app's
  "stream" composition mode emits `<label>.view.json` + `<label>.view.npz`
  (sorted per-file skip-ordinal arrays) into the output dir. The manifest may
  also carry **scrubs** — ordered regex rewrites `{name, field, pattern,
  replacement}` applied in-stream to kept records before extraction; exact
  per-scrub totals print at completion as `[scrub] name: docs=… subs=…
  chars_removed=…` lines (parsed by the web layer into the job result).
  With `"scrub_fixpoint": true` the chain repeats per record until a pass
  changes nothing (cap 10; `[scrub] fixpoint nonconverged=N` reports records
  that hit the cap). Scrubs — and the fixpoint flag when on — are part of
  the composition hash. Safety rails:
  each file's raw record count is verified at EOF (source drift since the view
  was cut is fatal, and the file is not marked done), and the composition's
  content hash is stamped to `<label>.manifest.view` — resuming the output dir
  with a different composition (or none) refuses rather than mixing records.
  The hash covers file list + counts + skip arrays, not manifest bytes, so a
  regenerated manifest with the identical composition resumes cleanly.
- **Shard writer**: writes `<label>_train_NNNNNN.npy` via a temp file +
  atomic `os.replace`; resumes at the next free index; stale `*.tmp-*` files
  from crashes are cleaned up on startup. One BOS is prepended per document
  (`add_bos=True`).
- **Token cache**: maintains a v2 `token_cache.json` one level above the output
  dir (shared across groups), with per-group token counts, shard hashes, and a
  fingerprint. Batch provenance is recorded under `source_batch` when input came
  from a Scriptorium batch.
- **Dependencies**: `numpy`, `zstandard`; `pyarrow` (lazy, Parquet only);
  `tokenizer_abstraction.get_tokenizer` from `../common_fsdp2`; optional
  `lang_detect`; `scanned-book-jsonl` lazily imports Scriptorium's
  `cleanup_ocr.cleanup_text` (from `/home/josef/valhalla/code/ocr`) for seam
  cleanup and falls back to a raw join if unavailable.
- **Multiprocessing**: uses the `spawn` context (Windows-safe). Workers ignore
  SIGINT; Ctrl-C in the parent hard-exits without flushing partial data to keep
  the dataset clean.
- **scanned-book-jsonl / batch**: emit exactly one joined document per book with
  per-page quality gating (defaults mirror Scriptorium's `ocr_quality.py`
  soft-fail thresholds); `body_only` is on unless `--include-matter`. `batch`
  mode resolves a frozen batch manifest's `output_abs` paths and processes each
  as a scanned-book JSONL, logging the batch's filter/collection/snapshot
  provenance.
- **Validation**: dies if `uint16` is requested but vocab ≥ 65536, or if no
  input files are found.
