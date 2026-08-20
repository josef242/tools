#!/usr/bin/env python3
"""
pre_tokenize.py
Unified pre-tokenisation script for LLM datasets.
Handles JSON, JSONL and Parquet; supports pluggable tokenizers; writes
NumPy shards of fixed token length.

Examples
--------
# Legacy tokenizer, JSONL files, template from --field
python pre_tokenize.py data_jsonl tokens --field text

# HuggingFace tokenizer, parquet, 100 M-token shards
python pre_tokenize.py parquet_dir tokens --input-format parquet \
       --tokenizer superbpe --tokenizer_path ./superbpe \
       --field body --shard-size 100000000
"""

from __future__ import annotations
import argparse, os, sys, glob, json, time, re, string, traceback, signal
import multiprocessing as mp, concurrent.futures as cf
from typing import Iterator, Sequence, Any
from datetime import datetime, timedelta

import numpy as np
import os, sys
import re, tempfile, uuid
import hashlib
try:
    import zstandard as zstd   # only needed for .jsonl.zst inputs
except ImportError:
    zstd = None
from pathlib import Path

# ------------------------- Common Files -------------------------
common_path = '../common_fsdp2'
if common_path not in sys.path:
    sys.path.insert(0, common_path)  # insert at the beginning to prioritize
# ---------------------------------------------------------------------------
# 1.  Prepare the tokenizer abstraction layer
# ---------------------------------------------------------------------------
from tokenizer_abstraction import get_tokenizer
from doc_shard_writer import DocShardWriter

# Language detection (add after other imports)
try:
    from lang_detect import is_english, LanguageDetector, LANG_DETECT_AVAILABLE
except ImportError:
    LANG_DETECT_AVAILABLE = False
    is_english = None
    LanguageDetector = None

# ---------------------------------------------------------------------------
# 2.  Nested formatter (from json_to_token_blobs)
# ---------------------------------------------------------------------------

class NestedFormatter(string.Formatter):
    _field_pattern = re.compile(
        r"(?:\.|^)(?P<name>[a-zA-Z_]\w*)|\[(?P<q>['\"])(?P<key>.+?)(?P=q)\]", re.VERBOSE)
    def get_field(self, field_name, args, kwargs):
        obj = kwargs
        for m in self._field_pattern.finditer(field_name):
            part = m.group("name") or m.group("key")
            try: obj = obj[part]
            # Show all of the keys in the error message
            except KeyError as ke:
                keys = ", ".join(f"'{k}'" for k in obj.keys())
                raise KeyError(f"Key '{part}' not found in {obj.__class__.__name__}; "
                             f"available keys: {keys}") from ke
            
        return obj, field_name

FMT = NestedFormatter()

# ---------------------------------------------------------------------------
# 3.  Dataset loaders
# ---------------------------------------------------------------------------

def iter_json_lines(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line: yield json.loads(line)

def iter_json_lines_zst(path: str) -> Iterator[dict]:
    import io
    if zstd is None:
        _die(f"{path}: .jsonl.zst input requires the 'zstandard' package (pip install zstandard)")
    with open(path, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for line in text_stream:
                line = line.strip()
                if line: yield json.loads(line)

def iter_json_array(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(f"{path} is JSON but not array")
        for rec in data: yield rec

def iter_scanned_book_jsonl(
    path: str,
    # Raised from 0.30 after the antonioycleopatr false-positive
    # (2026-04-23): mixed-language scholarly pages with embedded
    # foreign-language quotes + proper-noun-heavy bibliographies
    # routinely land at 0.30–0.50 even on clean OCR. 0.50 keeps
    # those pages; real gibberish OCR typically scores >0.7 so
    # this still filters garbage. Note this matches Stage 2a's
    # own hard-fail threshold (>0.5 at OCR time drops the page
    # entirely), so in practice anything still in the JSONL now
    # passes this gate — deliberately permissive at export time.
    max_non_dict_ratio: float = 0.50,
    min_alpha_ratio: float = 0.60,
    min_char_count: int = 150,
    max_repetition_ratio: float = 0.05,
    apply_seam_cleanup: bool = True,
    body_only: bool = True,
) -> Iterator[dict]:
    """Reader for Scriptorium's scanned-book JSONL format (v2).

    Format contract:
      - Row 1: {"record_type": "book_meta", ...book-level metadata...}
      - Rows 2..N: {"record_type": "page", "page": N, "text": str,
                    "quality": {char_count, non_dict_ratio,
                                mean_token_length, alpha_ratio,
                                repetition_ratio}}

    Yields exactly ONE dict per file:
      {"source": str, "title": str, "author": str, "language": str,
       "genre": str, "text": "<joined acceptable-page text>",
       "pages_used": int, "pages_dropped_by_filter": int}

    Page filtering re-derives the soft-fail decision from raw quality
    metrics using the threshold args (defaults match the Scriptorium
    ocr_quality.py soft-fail thresholds; override for strictness).
    One BOS per yielded doc (= one BOS per book) comes from the tokenizer.

    Seam cleanup: when apply_seam_cleanup=True (default), the joined
    text is passed through cleanup_ocr.cleanup_text with a narrow
    option set — remove_pages, join_sentences (sentence continuation
    across page breaks + hyphenated-word re-joining), and
    normalize_unicode_chars. We deliberately DO NOT apply dedupe_lines
    or front/back matter removal here — those are more aggressive
    and can touch bibliographic content in ways that surprised us
    empirically. Cost is ~1.5 MB/s serial on the main thread; for a
    500-book tier this is ~2–3 minutes total, negligible relative
    to tokenization.
    """
    # Lazy-import cleanup_text — avoids adding a Scriptorium dep to
    # tokenizer-only callers that don't need scanned-book-jsonl mode.
    cleanup_text = None
    if apply_seam_cleanup:
        import sys as _sys
        _sys.path.insert(0, "/home/josef/valhalla/code/ocr")
        try:
            from cleanup_ocr import cleanup_text  # type: ignore
        except ImportError:
            cleanup_text = None  # fall back to raw join
    header: dict = {}
    kept_texts: list[str] = []
    pages_used = 0
    pages_dropped = 0

    # Two-pass: first pull book_meta to see if a human override is
    # present. If so, compute body_only gating from override's page
    # range instead of the per-page matter_class flags (which reflect
    # 2b's algorithmic call, not the human correction). We only need
    # the header; streaming the whole file twice would be wasteful,
    # so we stash it on the first pass and reuse.
    override = None
    with open(path, "r", encoding="utf-8") as fh:
        first = fh.readline().strip()
        if first:
            try:
                header_preview = json.loads(first)
                if header_preview.get("record_type") == "book_meta":
                    override = header_preview.get("matter_boundaries_override")
            except json.JSONDecodeError:
                pass

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rtype = rec.get("record_type")
            if rtype == "book_meta":
                header = rec
                continue
            if rtype != "page":
                continue  # unknown row type; skip defensively
            # Stage 2b matter classification: skip front/back matter when
            # body_only=True (default). If the page hasn't been classified
            # yet (Stage 2b not run), matter_class is None and we keep the
            # page — legacy behavior for pre-2b JSONL.
            #
            # Override path: when `matter_boundaries_override` is on the
            # book_meta header, it's an authoritative human correction
            # over 2b's call. Use it to decide body-ness directly from
            # the page number rather than trusting per-page matter_class.
            if body_only:
                if override:
                    pg = rec.get("page")
                    in_body = (
                        pg is not None
                        and override.get("body_start_page") is not None
                        and override.get("body_end_page") is not None
                        and override["body_start_page"] <= pg <= override["body_end_page"]
                    )
                    if not in_body:
                        pages_dropped += 1
                        continue
                else:
                    mc = rec.get("matter_class")
                    if mc is not None and mc != "body":
                        pages_dropped += 1
                        continue
            q = rec.get("quality") or {}
            if q.get("char_count", 0) < min_char_count:
                pages_dropped += 1
                continue
            if q.get("non_dict_ratio", 0.0) > max_non_dict_ratio:
                pages_dropped += 1
                continue
            if q.get("alpha_ratio", 1.0) < min_alpha_ratio:
                pages_dropped += 1
                continue
            if q.get("repetition_ratio", 0.0) > max_repetition_ratio:
                pages_dropped += 1
                continue
            txt = rec.get("text") or ""
            if txt:
                kept_texts.append(txt)
                pages_used += 1

    joined = "\n\n".join(kept_texts)
    if joined and cleanup_text is not None:
        joined = cleanup_text(
            joined,
            remove_pages=True,
            join_sentences=True,
            normalize_unicode_chars=True,
            dedupe_lines=False,
            remove_front_matter=False,
            remove_back_matter=False,
            remove_fragments=False,
            remove_errors=False,
        )
    out = {
        "source": header.get("source"),
        "title": header.get("title"),
        "author": header.get("author"),
        "language": header.get("language"),
        "genre": header.get("genre"),
        "secondary_genres": header.get("secondary_genres"),
        "text": joined,
        "pages_used": pages_used,
        "pages_dropped_by_filter": pages_dropped,
    }
    # Skip empty books (every page filtered out)
    if joined:
        yield out


def read_scriptorium_batch(batch_path: str) -> tuple[dict, list[str]]:
    """Read a Scriptorium batch JSONL and return (meta, list of per-book
    v2 JSONL paths to process).

    Batch shape (from scriptorium's monitor/batch_writer.py):
      - Line 1: {"_batch_meta": {collection_id, filter_slug, slug, label,
                  created_at, row_count, filter_params, scriptorium_schema}}
      - Lines 2..N: {"source", "source_abs", "output_abs", ...ride-along
                     metadata including title/author/genre/language/quality...}

    A batch is essentially a frozen manifest: "these exact books at this
    point in time are what the training set was." pre_tokenize reads the
    file list, then processes each `output_abs` via the normal
    `iter_scanned_book_jsonl` path — the per-book v2 JSONL is what
    pre_tokenize already knows how to read, so no new iterator shape
    needed here.

    The `_batch_meta` header is returned so the caller can log provenance
    (what filter/collection/timestamp produced this training set).
    Non-JSON lines and rows missing `output_abs` are silently skipped —
    consistent with pre_tokenize's defensive style elsewhere.
    """
    meta: dict = {}
    paths: list[str] = []
    with open(batch_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_batch_meta" in rec:
                meta = rec["_batch_meta"] or {}
                continue
            out_abs = rec.get("output_abs")
            if out_abs:
                paths.append(out_abs)
    return meta, paths


def iter_parquet(path: str, text_col: str, batch_size: int) -> Iterator[dict]:
    import pyarrow.parquet as pq, pyarrow as pa
    pf = pq.ParquetFile(path)
    rowcount = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        rowcount += batch.num_rows
        # print(f"[{os.path.basename(path)}] read {rowcount:,} rows so far…", flush=True)
        table = pa.Table.from_batches([batch])
        if text_col not in table.column_names:
            # raise message, but also show columns
            list_of_columns = ", ".join(table.column_names)
            raise ValueError(f"Column {text_col} not found in {path}; "
                             f"available columns: {list_of_columns}")
            
        for txt in table[text_col].to_pylist():
            if isinstance(txt, str) and txt.strip():
                yield {text_col: txt}

# Auto-detect helper for JSON vs JSONL
def _detect_jsonl(path: str) -> bool:
    # read first 2 non-ws chars: '[' -> array json, '{' -> maybe jsonl
    with open(path, "r", encoding="utf-8") as fh:
        sample = "".join(ch for ch in fh.read(512) if not ch.isspace())  # strip ws
    return sample[:1] != '['

# ---------------------------------------------------------------------------
# 4.  CLI & helpers
# ---------------------------------------------------------------------------

def _die(msg:str):
    print(msg, file=sys.stderr); sys.exit(1)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-tokenise JSON/Parquet for LLM training")
    p.add_argument("input_dir", help="Dir or single file with dataset")
    p.add_argument("output_dir", help="Where .npy shards go")
    p.add_argument("--input-format",
                   choices=["auto","json","jsonl","parquet","scanned-book-jsonl","batch"],
                   default="auto",
                   help="Force format; default: auto. 'scanned-book-jsonl' expects "
                        "Scriptorium's v2 format (1 book-meta header + per-page rows); "
                        "emits ONE joined document per file with per-page quality filtering. "
                        "'batch' expects `input_dir` to be a Scriptorium batch JSONL file "
                        "(a frozen filter-match snapshot from `/batches` in the monitor) — "
                        "each row's `output_abs` is processed as a scanned-book-jsonl.")

    # Page-level quality thresholds for scanned-book-jsonl. Defaults mirror
    # the SOFT_FAIL_* constants in scriptorium/ocr_quality.py — at these values
    # pre_tokenize replicates the filter applied at OCR time. Override any
    # single value to change the per-page acceptance criteria WITHOUT having
    # to re-run Stage 2 OCR.
    sb = p.add_argument_group("scanned-book-jsonl page filter thresholds")
    sb.add_argument("--max-non-dict-ratio", type=float, default=0.50,
                    help="Drop pages with non_dict_ratio > THIS (default 0.50)")
    sb.add_argument("--min-alpha-ratio", type=float, default=0.60,
                    help="Drop pages with alpha_ratio < THIS (default 0.60)")
    sb.add_argument("--min-char-count", type=int, default=150,
                    help="Drop pages with char_count < THIS (default 150)")
    sb.add_argument("--max-repetition-ratio", type=float, default=0.05,
                    help="Drop pages with repetition_ratio > THIS (default 0.05)")
    sb.add_argument("--include-matter", action="store_true",
                    help="Include pages flagged by Stage 2b as front/back "
                         "matter. Default drops them (keeps only matter_class "
                         "in {'body', None}). Pages without matter_class "
                         "(Stage 2b never run) are always kept.")
    p.add_argument("--field", default="text",help="JSON key *or* parquet column containing the text (default: text). Ignored if --format is supplied.")
    p.add_argument("--format", dest="template",help="Full Python format string (overrides --field)")
    p.add_argument("--batch-size", type=int, default=1000,help="Rows per parquet batch (default 1000)")
    p.add_argument("--shard-size", type=int, default=int(1e8),help="Tokens per .npy shard (default 1e8)")
    p.add_argument("--val-holdout", type=int, default=50_000_000,
                   help="route the first ~N tokens of documents (arrival order ≈ corpus "
                        "head) to a *_val_* split (default 50M; 0 disables). Doc-aligned.")
    p.add_argument("--coprime", type=int, default=6,
                   help="force gcd(train shard count, this)=1 at close by splitting the "
                        "largest shard at a doc boundary (kills loader orbit-sharing; "
                        "6 covers world sizes 2^a*3^b)")
    p.add_argument("--min-shard", type=int, default=5_000_000,
                   help="merge a final shard smaller than this into its neighbor")
    p.add_argument("--legacy-river", action="store_true",
                   help="old behavior: fixed-size river shards that split documents at "
                        "shard boundaries, no val holdout, no manifest. Escape hatch only.")
    p.add_argument("--dtype", choices=["uint16", "uint32", "auto"],default="auto",help="Token dtype for .npy shards (uint16, uint32, or auto by vocab size)")
    p.add_argument("--label", default="data",help="Filename label prefix (default: data)")
    p.add_argument("--tokenizer", choices=["llama","hf","tiktoken","claude"], default="tiktoken")
    p.add_argument("--tokenizer_path", help="Path/ID for HF tokenizers")
    p.add_argument("--workers", type=int, default=max(mp.cpu_count()//2,1),help="Tokenisation worker processes")
    p.add_argument("--view-manifest",
                   help="Tokenize a COMPOSED view of the source (dataset-explorer "
                        "<label>.view.json): the manifest's file list is used verbatim "
                        "(no glob) and per-file record ordinals in its skip arrays are "
                        "dropped in-stream -- no intermediate copy. Each file's raw "
                        "record count is verified at EOF (source drift is fatal), and "
                        "an output dir started with a different composition refuses "
                        "to resume.")

    # Language filtering arguments
    lang_group = p.add_argument_group("Language filtering")
    lang_group.add_argument("--filter-english", action="store_true",
                           help="Filter to keep only English text")
    lang_group.add_argument("--lang-threshold", type=float, default=0.8,
                           help="Confidence threshold for language detection (0-1, default: 0.8)")
    lang_group.add_argument("--lang-backend", choices=["auto", "fasttext", "langdetect", "langid"],
                           default="auto", help="Language detection backend")
    lang_group.add_argument("--lang-model", help="Path to FastText language model (if using fasttext)")
    lang_group.add_argument("--lang-sample-size", type=int, default=500,
                           help="Max characters to sample for language detection (default: 500)")

    return p.parse_args()

# ---------------------------------------------------------------------------
# 5.  Shard writer
# ---------------------------------------------------------------------------

class ShardWriter:
    """
    Accumulates tokens and writes fixed-size NumPy shards.
    • Writes to <label>_train_000123.npy.tmp first;
      when the write finishes it atomically renames to .npy.
    • On resume, starts at the next index after the highest .npy file.
    • Any stray *.tmp files from earlier crashes are silently removed.
    """
    _fname_re = re.compile(r"(.+)_train_(\d{6})\.npy$")

    def __init__(self, outdir: str, label: str, shard_size: int, dtype: str = "uint16"):
        self.outdir, self.label, self.size = outdir, label, shard_size
        os.makedirs(outdir, exist_ok=True)
        self._cleanup_tmp()

        self.index = self._next_index()
        np_dtype = np.uint16 if dtype == "uint16" else np.uint32
        self.buf   = np.empty((shard_size,), dtype=np_dtype)
        self.used  = 0

    # ── public API ──────────────────────────────────────────────────────
    def add(self, arr: np.ndarray):
        start = 0
        while start < len(arr):
            room = self.size - self.used
            take = min(room, len(arr) - start)
            self.buf[self.used:self.used+take] = arr[start:start+take]
            self.used += take
            start     += take
            if self.used == self.size:
                self._flush()

    def close(self):
        if self.used:
            self._flush()

    # ── helpers ─────────────────────────────────────────────────────────
    def _flush(self):
        tmp_path  = self._tmp_name()
        final_path = self._final_name()

        # write to .tmp
        np.save(tmp_path, self.buf[:self.used])
        # atomic move ⇒ either old file stays untouched or new one fully appears
        os.replace(tmp_path+".npy", final_path)

        print(f"[write] {final_path}  ({self.used:,} tokens)", flush=True)
        self.index += 1
        self.used   = 0

    # create deterministic filenames ------------------------------------
    def _final_name(self) -> str:
        return os.path.join(
            self.outdir, f"{self.label}_train_{self.index:06d}.npy")

    def _tmp_name(self) -> str:
        # keep same prefix for easy cleanup + a UUID to avoid clashes
        return self._final_name() + f".tmp-{uuid.uuid4().hex}"

    # resume logic -------------------------------------------------------
    def _next_index(self) -> int:
        max_idx = -1
        for fname in os.listdir(self.outdir):
            m = self._fname_re.match(fname)
            if m and m.group(1) == self.label:
                max_idx = max(max_idx, int(m.group(2)))
        return max_idx + 1

    def _cleanup_tmp(self):
        for fname in os.listdir(self.outdir):
            if ".tmp-" in fname:                      # our temp pattern
                try:
                    os.remove(os.path.join(self.outdir, fname))
                    print(f"[cleanup] removed stale temp file {fname}", flush=True)
                except OSError:
                    pass

class FilterStats:
    """Track filtering statistics."""
    def __init__(self):
        self.total_docs = 0
        self.kept_docs = 0
        self.filtered_docs = 0
        self.lang_errors = 0
        
    def add_doc(self, kept: bool, error: bool = False):
        self.total_docs += 1
        if error:
            self.lang_errors += 1
        elif kept:
            self.kept_docs += 1
        else:
            self.filtered_docs += 1
    
    def summary(self) -> str:
        if self.total_docs == 0:
            return "No documents processed"
        
        kept_pct = (self.kept_docs / self.total_docs) * 100
        filtered_pct = (self.filtered_docs / self.total_docs) * 100
        error_pct = (self.lang_errors / self.total_docs) * 100
        
        return (
            f"Language filtering stats:\n"
            f"  Total documents: {self.total_docs:,}\n"
            f"  Kept (English): {self.kept_docs:,} ({kept_pct:.1f}%)\n"
            f"  Filtered (non-English): {self.filtered_docs:,} ({filtered_pct:.1f}%)\n"
            f"  Detection errors: {self.lang_errors:,} ({error_pct:.1f}%)"
        )

# ---------------------------------------------------------------------------
# 6.  Worker initialiser / function
# ---------------------------------------------------------------------------

def _init_worker(tok_name, tok_path, dtype):
    global TOK, USE_UINT16
    TOK = get_tokenizer(tok_name, tok_path)
    USE_UINT16 = (dtype == "uint16")

def _worker_sigint_ignore():
    """Make worker ignore Ctrl-C; only the parent handles it."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def _init_worker_wrapper(tok_name, tok_path, dtype):
    """Pool initializer: set SIGINT handler then build tokenizer."""
    _worker_sigint_ignore()
    _init_worker(tok_name, tok_path, dtype)     # ← existing function

def _tokenize(doc_tpl: tuple[str, str]) -> np.ndarray:
    text, template = doc_tpl
    if USE_UINT16:
        arr = TOK.encode_to_uint16(text, add_bos=True)
    else:
        arr = TOK.encode_to_uint32(text, add_bos=True)
    return arr

# ---------------------------------------------------------------------------
# 6.  Helper: write finished jobs
# ---------------------------------------------------------------------------
def _flush_done(futs: set[cf.Future], shard: "ShardWriter") -> int:
    """Write completed futures to *shard*.  
    Returns the number of tokens written."""
    tokens = 0
    for fut in futs:
        arr = fut.result()           # re-raise worker exceptions if any
        shard.add(arr)
        tokens += len(arr)
    return tokens

def scan_group_shards(data_root_path, group_name, split="train"):
    """Scan all shards for a group and return metadata"""
    # dir_path = os.path.join(data_root_path, group_name)
    pattern = f"*_{split}_*.npy"
    shards = sorted(glob.glob(os.path.join(data_root_path, pattern)))
    
    if not shards:
        print(f"  WARNING: No shards found for {group_name} in {data_root_path}")
        return None
    
    print(f"  Scanning {group_name}: {len(shards)} shards...", end='', flush=True)
    
    total_tokens = 0
    all_shard_stats = []  # Collect ALL file stats for fingerprinting
    
    for i, shard_path in enumerate(shards):
        if i % 10 == 0:
            print(f"\r  Scanning {group_name}: {i}/{len(shards)} shards...", end='', flush=True)
        
        # Get file stats (fast)
        stat = os.stat(shard_path)
        shard_info = {
            'name': os.path.basename(shard_path),
            'size': stat.st_size,
            'mtime': int(stat.st_mtime)  # Convert to int for consistent hashing
        }
        all_shard_stats.append(shard_info)
        
        # Only memory-map for token counting
        arr = np.load(shard_path, mmap_mode='r')
        total_tokens += arr.shape[0]
        
        # Store detailed metadata for first few shards (for debugging/verification)
        if i < 3:
            shard_info['tokens'] = int(arr.shape[0])
            shard_info['dtype'] = str(arr.dtype)
    
    print(f"\r  {group_name}: {len(shards)} shards, {total_tokens/1e9:.2f}B tokens")
    
    return {
        'token_count': total_tokens,
        'shard_count': len(shards),
        'all_shards_hash': compute_shards_hash(all_shard_stats),  # Hash of ALL files
        'sample_shards': all_shard_stats[:3]  # Keep for debugging/verification
    }

def compute_shards_hash(shard_stats):
    """Compute hash of all shard metadata"""
    hasher = hashlib.sha256()
    
    # Include all files in deterministic order
    for shard in shard_stats:
        hasher.update(f"{shard['name']},{shard['size']},{shard['mtime']}".encode())
    
    return hasher.hexdigest()[:16]  # Just first 16 chars is plenty

def compute_fingerprint(groups_metadata):
    """Create a fingerprint from the groups metadata"""
    hasher = hashlib.sha256()
    
    # Sort groups for consistent hashing
    for group_name in sorted(groups_metadata.keys()):
        meta = groups_metadata[group_name]
        hasher.update(group_name.encode())
        hasher.update(str(meta['shard_count']).encode())
        hasher.update(meta['all_shards_hash'].encode())  # This covers ALL files
    
    return hasher.hexdigest()

# ---------------------------------------------------------------------------
# 7.  Manifest management with token counts
# ---------------------------------------------------------------------------
def normalize_path(path_str):
    """Normalize a path for consistent comparison across platforms"""
    return str(Path(path_str))

def load_manifest_with_tokens(manifest_path):
    """Load manifest and return (processed_files set, total_tokens from previous runs)"""
    processed_files = set()
    previous_tokens = 0
    
    if not os.path.exists(manifest_path):
        return processed_files, previous_tokens
        
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # Check if it's the new format with token count
            if '\t' in line:
                filepath, token_count = line.split('\t', 1)
                # Normalize the path when loading
                processed_files.add(normalize_path(filepath))
                previous_tokens += int(token_count)
            else:
                # Old format - just filename
                processed_files.add(normalize_path(line))
                # We don't know the token count for old entries
                
    return processed_files, previous_tokens

def append_to_manifest(manifest_path, filepath, token_count):
    """Append a file and its token count to the manifest"""
    with open(manifest_path, "a", encoding="utf-8") as fh:
        # Store the original path string to maintain readability
        fh.write(f"{filepath}\t{token_count}\n")

# ---------------------------------------------------------------------------
# 7b.  View manifests (dataset-explorer composed views)
# ---------------------------------------------------------------------------

def load_view_manifest(path):
    """Load a dataset-explorer view manifest: a pointer-tier description of a
    composed dataset -- the authoritative source file list plus per-file
    ordinals of records to SKIP (sorted int arrays in a sibling .npz).
    Attaches '_skips'/'_expected' lookups (normalized path -> array / raw
    record count) and '_hash', a content hash over (files, counts, skip
    arrays) -- deliberately not over the manifest bytes, so a regenerated
    manifest with the identical composition still resumes."""
    with open(path, "r", encoding="utf-8") as fh:
        v = json.load(fh)
    npz = None
    if v.get("skips"):
        npz_path = os.path.join(os.path.dirname(os.path.abspath(path)), v["skips"])
        npz = np.load(npz_path)
    h = hashlib.sha256()
    v["_skips"], v["_expected"] = {}, {}
    for i, e in enumerate(v["files"]):
        key = normalize_path(e["path"])
        name = f"skip_{i:05d}"
        arr = (np.asarray(npz[name], dtype=np.int64)
               if npz is not None and name in npz.files
               else np.empty(0, dtype=np.int64))
        v["_skips"][key] = arr
        v["_expected"][key] = int(e["records"])
        h.update(key.encode()); h.update(str(e["records"]).encode())
        h.update(arr.tobytes())
    # Scrubs: ordered regex rewrites applied in-stream to KEPT records.
    # They are part of the composition's identity, so they join the hash --
    # editing a scrub and resuming into the same output dir must refuse.
    scrubs = []
    for s in (v.get("scrubs") or []):
        try:
            pat = re.compile(s["pattern"])
        except re.error as e:
            _die(f"[view] scrub {s.get('name')!r}: bad regex: {e}")
        scrubs.append((s["name"], s["field"], pat, s.get("replacement", "")))
    v["_scrubs"] = scrubs
    v["_scrub_fixpoint"] = bool(v.get("scrub_fixpoint"))
    h.update(json.dumps(v.get("scrubs") or [], sort_keys=True,
                        ensure_ascii=False).encode())
    if v["_scrub_fixpoint"]:
        # Only hashed when ON so pre-fixpoint manifests keep their old hash
        # (a False flag must not invalidate existing resume stamps).
        h.update(b"scrub_fixpoint")
    v["_hash"] = h.hexdigest()
    return v


def check_view_stamp(manifest_path, view, resuming: bool):
    """Resume guard: a view run stamps its composition hash next to the resume
    manifest. Resuming with a different (or no) view would silently mix two
    compositions in one shard set -- refuse instead."""
    stamp_path = manifest_path + ".view"
    stamp = None
    if os.path.exists(stamp_path):
        with open(stamp_path, "r", encoding="utf-8") as fh:
            stamp = fh.read().strip()
    if view is None:
        if stamp is not None:
            _die(f"Output dir was started with --view-manifest ({stamp_path} exists) "
                 f"but this run has none -- resuming would mix compositions.")
        return
    if stamp is not None:
        if stamp != view["_hash"]:
            _die("--view-manifest composition differs from the one this output dir "
                 "was started with -- resuming would mix compositions. Use a fresh "
                 "output dir, or delete the manifest + .view stamp to start over.")
    elif resuming:
        _die(f"Output dir has an existing {os.path.basename(manifest_path)} from a "
             f"non-view run -- refusing to resume it with --view-manifest.")
    else:
        os.makedirs(os.path.dirname(stamp_path) or ".", exist_ok=True)
        with open(stamp_path, "w", encoding="utf-8") as fh:
            fh.write(view["_hash"] + "\n")


SCRUB_MAX_PASSES = 10       # fixpoint cap (mirrors the explorer web layer)


def _apply_view(iterator, skip, expected, path, scrubs=(), scrub_counts=None,
                fixpoint=False, nonconverged=None):
    """Stream `iterator`, dropping records whose ordinal is in sorted `skip`,
    then applying the view's scrubs (ordered regex rewrites) to survivors --
    once, or to fixpoint (chain repeated until a pass changes nothing, capped)
    when the manifest asks for it. Verifies the raw record count at EOF: a
    mismatch means the source changed since the view was cut, every ordinal
    after the drift point is suspect, and the file must NOT be marked done --
    so it's fatal."""
    si, sn = 0, len(skip)
    n = 0
    for rec in iterator:
        ordinal = n
        n += 1
        if si < sn and skip[si] == ordinal:
            si += 1
            continue
        if scrubs:
            hit_names = set()
            passes = 0
            while True:
                passes += 1
                pass_changed = False
                for name, field, pat, repl in scrubs:
                    v = rec.get(field)
                    if not isinstance(v, str):
                        continue
                    new, hits = pat.subn(repl, v)
                    if hits:
                        rec[field] = new
                        pass_changed = True
                        if scrub_counts is not None:
                            c = scrub_counts.setdefault(name, [0, 0, 0])
                            if name not in hit_names:
                                c[0] += 1
                            c[1] += hits
                            c[2] += len(v) - len(new)
                        hit_names.add(name)
                if not pass_changed or not fixpoint:
                    break
                if passes >= SCRUB_MAX_PASSES:
                    if nonconverged is not None:
                        nonconverged[0] += 1
                    break
        yield rec
    if n != expected:
        raise RuntimeError(
            f"[view] {path}: raw record count {n:,} != view manifest's {expected:,} "
            f"-- source changed since the view was cut; aborting")

# ---------------------------------------------------------------------------
# 8.  Main
# ---------------------------------------------------------------------------
def main() -> None:
    args      = parse_args()

     # Language detection setup
    lang_detector = None
    filter_stats = FilterStats()
    
    if args.filter_english:
        if not LANG_DETECT_AVAILABLE:
            _die("--filter-english requires a language detection library. "
                 "Install with: pip install langdetect (or fasttext-wheel for better performance)")
        
        try:
            backend = None if args.lang_backend == "auto" else args.lang_backend
            lang_detector = LanguageDetector(backend=backend, model_path=args.lang_model)
            print(f"[config] Language filtering enabled (backend: {lang_detector.backend}, threshold: {args.lang_threshold})")
        except Exception as e:
            _die(f"Failed to initialize language detector: {e}")

    # decide the dtype for the shards
    tok = get_tokenizer(args.tokenizer, args.tokenizer_path)   # new
    vocab_size = len(tok)

    if args.dtype == "auto":
        dtype = "uint16" if vocab_size < 65_536 else "uint32"
    else:
        dtype = args.dtype

    if dtype == "uint16" and vocab_size >= 65_536:
        _die("dtype uint16 requested but vocab ≥ 65 536")
    print(f"[config] using {dtype} shards")

    if dtype == "uint16" and vocab_size >= 65_536:
        _die(f"dtype uint16 requested but vocab size is {vocab_size:,} (>= 65 536)")


    template  = args.template or f"{{{args.field}}}"
    formatter = FMT

    # Progress Manifest - now with token counts
    MANIFEST = os.path.join(args.output_dir, f"{args.label}.manifest")
    print(f"[config] Manifest file: {MANIFEST}")
    print(f"Checking for previously processed files...")
    processed_files, previous_tokens = load_manifest_with_tokens(MANIFEST)

    if processed_files:
        print(f"  Found {len(processed_files)} previously processed files in manifest")
        print(f"  Previous runs processed {previous_tokens:,} tokens")
    else:
        print("  No previous manifest found or it's empty; starting fresh")
    
    # Start with tokens from previous runs
    token_total = previous_tokens

    # ── gather files ─────────────────────────────────────────────────────
    batch_meta: dict | None = None
    if args.input_format == "batch":
        # `input_dir` is actually a batch JSONL file path. Resolve it to a
        # list of per-book v2 JSONL paths via the batch's rows, and log
        # the batch's provenance so stdout shows exactly which filter /
        # collection / snapshot timestamp produced this training set.
        if not os.path.isfile(args.input_dir):
            _die(f"--input-format batch expects a batch JSONL file path; "
                 f"got: {args.input_dir}")
        print(f"Reading batch manifest {args.input_dir}…")
        batch_meta, files = read_scriptorium_batch(args.input_dir)
        print(f"[batch] slug:        {batch_meta.get('slug') or '?'}")
        print(f"[batch] label:       {batch_meta.get('label') or '?'}")
        print(f"[batch] filter:      {batch_meta.get('filter_slug') or '(ad-hoc)'}")
        print(f"[batch] collection:  {batch_meta.get('collection_id') or '?'}")
        print(f"[batch] snapshotted: {batch_meta.get('created_at') or '?'}")
        print(f"[batch] schema v{batch_meta.get('scriptorium_schema') or '?'} · "
              f"{len(files):,} books in manifest")
        # Any output_abs missing on disk is a signal that Stage 2a
        # hasn't completed for that book (or a path rename since the
        # batch was frozen). Fail-soft — drop them with a visible count
        # so the operator knows, rather than silently skipping or hard-
        # erroring.
        existing = [p for p in files if os.path.isfile(p)]
        missing = len(files) - len(existing)
        if missing:
            print(f"[batch] WARNING: {missing:,} of {len(files):,} output paths "
                  f"missing on disk — Stage 2a may not have completed those books "
                  f"since the batch was snapshotted. Proceeding with {len(existing):,}.")
        files = existing
    else:
        print(f"Scanning input dir {args.input_dir} (format: {args.input_format})…")
        if os.path.isfile(args.input_dir):
            files = [args.input_dir]
        else:
            patterns = []
            if args.input_format in ("auto", "parquet"):
                patterns.append("**/*.parquet")
            if args.input_format in ("auto", "jsonl", "scanned-book-jsonl"):
                patterns.append("**/*.jsonl")
                patterns.append("**/*.jsonl.zst")
            if args.input_format in ("auto", "json"):
                patterns.append("**/*.json")

            files = []
            for pat in patterns:
                files += glob.glob(os.path.join(args.input_dir, pat), recursive=True)

    # ── composed view (dataset-explorer bridge) ──────────────────────────
    view = None
    if args.view_manifest:
        if args.input_format in ("scanned-book-jsonl", "batch"):
            _die("--view-manifest cannot be combined with scanned-book-jsonl/batch "
                 "input (those iterators renumber records, so the view's ordinals "
                 "would not correspond)")
        view = load_view_manifest(args.view_manifest)
        # The manifest's file list is AUTHORITATIVE: skip ordinals were computed
        # against exactly these files, and a glob here could disagree (e.g.
        # .json sidecars in mixed dirs) and silently misalign every set index.
        files = [e["path"] for e in view["files"]]
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            _die(f"[view] {len(missing)} of {len(files)} file(s) in the view "
                 f"manifest missing on disk, e.g. {missing[0]}")
        print(f"[view] {os.path.basename(args.view_manifest)}: {len(files)} file(s); "
              f"skipping {view.get('dropped', 0):,} of "
              f"{view.get('kept', 0) + view.get('dropped', 0):,} records in-stream"
              + (f"; {len(view['_scrubs'])} scrub(s)" if view["_scrubs"] else ""))
    view_scrub_counts: dict = {}
    view_nonconverged = [0]
    check_view_stamp(MANIFEST, view, resuming=bool(processed_files))

    if not files:
        _die("No input files found")

    print(f"Found {len(files)} file(s); spawning {args.workers} workers…")
    # The writer's resume scan validates every pre-existing shard in the output
    # dir before anything else happens -- minutes on a large tree over a network
    # mount, and previously totally silent.
    print("[writer] checking existing shards in output dir "
          "(can take a while on network storage)…", flush=True)
    t0          = time.time()
    if args.legacy_river:
        shard = ShardWriter(args.output_dir, args.label, args.shard_size, dtype)
    else:
        # Doc-aligned writer (2026-07-14): whole documents per shard, val holdout
        # from the stream head, coprime shard count, rechunk-schema manifest.
        # BOS comes from the tokenizer object — never a literal (bos-audit rule).
        shard = DocShardWriter(
            args.output_dir, args.label, bos_id=tok.bos_id,
            cap=args.shard_size, min_shard=args.min_shard,
            coprime=args.coprime, val_holdout=args.val_holdout, dtype=dtype)

    # ── pool (spawn on Windows) ──────────────────────────────────────────
    ctx = mp.get_context("spawn")

    with cf.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=_init_worker_wrapper,
            initargs=(args.tokenizer, args.tokenizer_path, dtype)) as pool:

        MAX_QUEUED = args.workers * 4     # outstanding jobs per worker
        HEARTBEAT  = 50_000               # log every N docs
        pending: list[cf.Future] = []

        try:
            
            total_files     = len(files)
            # Start with count of already processed files
            # Normalize paths for comparison
            normalized_files = {normalize_path(f): f for f in files}
            completed_files = len([f for f in normalized_files if f in processed_files])
            initial_completed = completed_files  # Track for ETA calculation

            # Add resume summary here
            already_done = [orig_path for norm_path, orig_path in normalized_files.items() 
                           if norm_path in processed_files]
            if already_done:
                print(f"[resume] Skipping {len(already_done)} already-processed files from manifest")
                print(f"[resume] Previous runs processed {previous_tokens:,} tokens")
                remaining = [orig_path for norm_path, orig_path in normalized_files.items() 
                            if norm_path not in processed_files]
                print(f"[resume] Will process {len(remaining)} remaining files")
                print("=" * 60)

            for path in files:
                normalized_path = normalize_path(path)
                if normalized_path in processed_files:
                    continue   # already done last run
                    
                # Track tokens for this file
                file_token_count = 0
                    
                # choose iterator for this file ---------------------------
                # Batch mode re-uses scanned-book-jsonl — the batch just
                # supplied the file list, each entry is still a per-book
                # v2 JSONL that the iterator already knows.
                ext = os.path.splitext(path)[1].lower()
                if args.input_format in ("scanned-book-jsonl", "batch"):
                    iterator = iter_scanned_book_jsonl(
                        path,
                        max_non_dict_ratio=args.max_non_dict_ratio,
                        min_alpha_ratio=args.min_alpha_ratio,
                        min_char_count=args.min_char_count,
                        max_repetition_ratio=args.max_repetition_ratio,
                        body_only=not args.include_matter,
                    )
                elif path.endswith(".jsonl.zst"):
                    iterator = iter_json_lines_zst(path)
                elif args.input_format == "parquet" or ext == ".parquet":
                    iterator = iter_parquet(path, args.field, args.batch_size)
                else:
                    if (args.input_format == "jsonl" or ext == ".jsonl"
                            or (args.input_format == "auto" and _detect_jsonl(path))):
                        iterator = iter_json_lines(path)
                    else:
                        iterator = iter_json_array(path)

                if view is not None:
                    key = normalize_path(path)
                    iterator = _apply_view(iterator, view["_skips"][key],
                                           view["_expected"][key], path,
                                           scrubs=view["_scrubs"],
                                           scrub_counts=view_scrub_counts,
                                           fixpoint=view["_scrub_fixpoint"],
                                           nonconverged=view_nonconverged)

                rows_seen = 0
                for rec in iterator:
                    try:
                        txt = formatter.format(template, **rec).strip()
                        if not txt:
                            continue
                    except KeyError as ke:
                        print(f"[skip] {ke} in {path}")
                        continue

                    # Language filtering
                    if lang_detector is not None:
                        try:
                            # Sample first N characters for efficiency
                            sample = txt[:args.lang_sample_size]
                            if not lang_detector.is_english(sample, threshold=args.lang_threshold):
                                #print(f"[filter] Non-English text detected in {path}, skipping")
                                # display sample
                                #print(f"[filter] Sample: {sample[:100]}{'...' if len(sample) > 100 else ''}")
                                filter_stats.add_doc(kept=False)
                                continue
                            filter_stats.add_doc(kept=True)
                        except Exception as e:
                            # On error, keep the document and count the error
                            filter_stats.add_doc(kept=True, error=True)
                            if filter_stats.lang_errors == 1:  # Only log first error
                                print(f"[warning] Language detection error: {e}")

                    # submit tokenisation job
                    pending.append(pool.submit(_tokenize, (txt, template)))
                    rows_seen += 1

                    if rows_seen % HEARTBEAT == 0:
                        print(f"[{os.path.basename(path)}] queued {rows_seen:,} docs",
                              flush=True)

                    # throttle queue size
                    if len(pending) >= MAX_QUEUED:
                        done, still_pending = cf.wait(
                            pending, return_when=cf.FIRST_COMPLETED)
                        tokens = _flush_done(done, shard)
                        token_total += tokens
                        file_token_count += tokens
                        pending = list(still_pending)    # keep list semantics

                # drain whatever's left for this file
                done, still_pending = cf.wait(
                    pending, timeout=0, return_when=cf.ALL_COMPLETED)
                tokens = _flush_done(done, shard)
                token_total += tokens
                file_token_count += tokens
                pending = list(still_pending)
            
                # ▼───────────────────────────────────────────────────────────────────
                # File finished successfully → append to manifest with token count
                append_to_manifest(MANIFEST, path, file_token_count)
                processed_files.add(normalized_path)
                # ▲───────────────────────────────────────────────────────────────────

                completed_files += 1
                elapsed = time.time() - t0
                # Calculate ETA based only on files processed in this run
                files_done_this_run = completed_files - initial_completed
                if files_done_this_run > 0:
                    eta = (elapsed / files_done_this_run) * (total_files - completed_files)
                else:
                    eta = 0
                    
                print(f"=== {completed_files}/{total_files} files "
                    f"({completed_files/total_files:.1%}) • "
                    f"{token_total:,} tokens (file: {file_token_count:,}) • "
                    # Info about filtering
                    f"filtered: {filter_stats.filtered_docs:,} • "
                    f"ETA {timedelta(seconds=int(eta))} ===", flush=True)
                
        except KeyboardInterrupt:
            print("\n[Ctrl-C] stopping…", flush=True)
            pool.shutdown(cancel_futures=True, wait=False)
            for p in mp.active_children():
                p.kill()
            # Do **not** flush; keep any partial data out of the dataset
            os._exit(0)          # hard‑exit, skipping the ctx‑manager's shutdown
        else:
            # Final drain: any pending futures from the last files. The
            # per-file drain above uses timeout=0 (non-blocking poll) which
            # works for formats that yield many docs per file but misses
            # the lone submission from scanned-book-jsonl (1 doc per file).
            # Without this drain, those docs never land in a shard.
            if pending:
                done, _ = cf.wait(pending, return_when=cf.ALL_COMPLETED)
                tokens = _flush_done(done, shard)
                token_total += tokens
                print(f"[final drain] {tokens:,} tokens, {token_total:,} total",
                      flush=True)
            shard.close()           # normal exit, no Ctrl-C
            # Exact per-scrub totals for the ledger; the explorer web layer
            # parses these lines into the composed-job result.
            if view is not None and view["_scrubs"]:
                for sname, _sf, _sp, _sr in view["_scrubs"]:
                    c = view_scrub_counts.get(sname, [0, 0, 0])
                    print(f"[scrub] {sname}: docs={c[0]:,} subs={c[1]:,} "
                          f"chars_removed={c[2]:,}", flush=True)
                if view["_scrub_fixpoint"]:
                    print(f"[scrub] fixpoint nonconverged={view_nonconverged[0]:,}",
                          flush=True)

        finally:
            pass

        # At the end of main(), after shard.close():

    # Generate or update the cache file
    # Token cache file goes up one level, so it can be shared across groups
    parent_dir = os.path.join(args.output_dir, "..")
    cache_path = os.path.join(parent_dir, "token_cache.json")
    existing_cache = {}

    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            existing_cache = json.load(f)

    # Scan what we just created
    print(f"\nUpdating token cache for {args.label}...")
    metadata = scan_group_shards(args.output_dir, args.label)

    if metadata is None:
        print(f"  WARNING: No shards found for {args.label}, skipping cache update.")
        return

    # Record batch provenance on this group's metadata when the input
    # came from a Scriptorium batch — lets a future reader of the cache
    # answer "which filter/collection/snapshot produced this dataset?"
    # without having to re-open the batch JSONL. Only the stable fields
    # are kept; `filter_params` is omitted to keep the cache compact.
    if batch_meta:
        metadata["source_batch"] = {
            "collection_id": batch_meta.get("collection_id"),
            "filter_slug":   batch_meta.get("filter_slug"),
            "slug":          batch_meta.get("slug"),
            "label":         batch_meta.get("label"),
            "created_at":    batch_meta.get("created_at"),
            "row_count":     batch_meta.get("row_count"),
            "scriptorium_schema": batch_meta.get("scriptorium_schema"),
        }

    # Determine tokenizer from args
    tokenizer_name = args.tokenizer_path if args.tokenizer == "hf" else args.tokenizer

    # Update or create cache structure
    if "version" in existing_cache and existing_cache["version"] == "2.0":
        # Update existing v2 cache
        if "groups" not in existing_cache:
            existing_cache["groups"] = {}
        existing_cache["groups"][args.label] = metadata
        existing_cache[args.label] = metadata['token_count']  # Backward compat
        
        # Recompute fingerprint
        existing_cache["fingerprint"] = compute_fingerprint(existing_cache["groups"])
        existing_cache["updated_at"] = datetime.now().isoformat()
    else:
        # Create new v2 cache or upgrade v1
        new_cache = {
            "version": "2.0",
            "generated_by": "pre_tokenize.py",
            "generated_at": datetime.now().isoformat(),
            "tokenizer": tokenizer_name,
            "data_root": args.output_dir,
            "groups": {args.label: metadata},
            args.label: metadata['token_count']  # Backward compat
        }
        
        # Preserve other groups from v1 cache
        for key, value in existing_cache.items():
            if isinstance(value, int) and key != args.label:
                # Migrate v1 token counts
                new_cache[key] = value
                # Create minimal metadata
                new_cache["groups"][key] = {"token_count": value}
        
        new_cache["fingerprint"] = compute_fingerprint(new_cache["groups"])
        existing_cache = new_cache

    # Save updated cache
    with open(cache_path, 'w') as f:
        json.dump(existing_cache, f, indent=2)

    print(f"Updated cache: {cache_path}")
    print(f"  {args.label}: {metadata['token_count']/1e9:.2f}B tokens")

    # ── summary ──────────────────────────────────────────────────────────
    if lang_detector is not None:
        print("=" * 60)
        print(filter_stats.summary())
        
    dt = time.time() - t0
    tokens_this_run = token_total - previous_tokens
    print("=" * 60)
    # '->' not '→': a Windows cp1252 stdout pipe cannot encode the arrow,
    # and this line once crashed the process AFTER 21.9B tokens of successful
    # work -- the most expensive character in the codebase's history.
    print(f"Done. {token_total:,} total tokens -> {args.output_dir}")
    print(f"  Previous runs: {previous_tokens:,} tokens")
    print(f"  This run: {tokens_this_run:,} tokens in {timedelta(seconds=int(dt))}")
    print(f"  ({tokens_this_run / dt:,.1f} tok/s) "
          f"({len(files) - len(already_done) if 'already_done' in locals() else len(files)} files processed, {args.workers} workers) "
          f"dtype: {dtype}")

if __name__ == "__main__":
    main()