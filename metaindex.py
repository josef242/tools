#!/usr/bin/env python3
"""Columnar metadata index over a JSONL / JSONL.zst corpus, for interactive curation.

Scanning millions of JSON records per query is unusable interactively -- you would wait
minutes to discover a query had a typo. This builds a parquet index ONCE, after which
queries are vectorized column reads: milliseconds, and only the columns a query actually
mentions are read from disk.

TEXT IS NEVER INDEXED. Only top-level scalar fields and the flattened `metadata` dict are
stored; the document body is deliberately excluded (see EXCLUDE_FIELDS). That keeps the
index small -- metadata is a rounding error next to the corpus -- and means curating a
corpus never requires materializing its contents.

Row order IS global record order, so row i of the index is record i in the explorer. That
invariant is what lets a query result drop straight into the existing search state and be
navigated with list / next / goto, with no separate plumbing.
"""

import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False

# Never indexed: the document body. Curation needs metadata, not content.
EXCLUDE_FIELDS = {'text', 'body', 'content', 'raw_content', 'document'}

# Records scanned to discover the schema before writing begins. Parquet needs a fixed
# schema per file, and buffering every row to learn it would cost gigabytes at corpus
# scale, so we learn from a prefix and report anything that turns up later.
# Fields this module CREATES, as opposed to ones it read from the corpus. Listed so
# discovery can declare them without counting, and so `meta fields` can tell the user
# which keys exist only because we computed them.
DERIVED_FIELD_DOC = {
    'words':          'declared count if present, else counted from the body',
    'words_actual':   'always present -- counted from the body at index time',
    'words_declared': 'the value the corpus declared; empty where it omitted one',
    'words_ratio':    'words_actual / words_declared; blank if none was declared',
}
DERIVED_FIELDS = tuple(DERIVED_FIELD_DOC)

DISCOVER_RECORDS = 200_000
# Adaptive discovery: stop early once the schema stabilizes -- no new field for
# DISCOVER_STABLE consecutive records, after at least DISCOVER_MIN records. On a
# uniform corpus this turns a near-full silent pass (200k BOOKS, once) into a
# few seconds; late-appearing rare fields still surface via the unknown_fields
# warning and a 'meta rebuild'.
DISCOVER_MIN = 1_000
DISCOVER_STABLE = 2_000
CHUNK_ROWS = 100_000

# Mean characters above which an indexed column is flagged as probable body text.
# AO3 tag lists reach a few hundred characters; prose starts in the thousands.
LONG_VALUE_CHARS = 1500


def open_text(path: Path):
    """Text stream over .jsonl or .jsonl.zst."""
    if path.name.lower().endswith('.zst'):
        import zstandard as zstd
        return io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(open(path, 'rb')),
                                encoding='utf-8', errors='replace')
    return open(path, 'r', encoding='utf-8', errors='replace')


def _as_text(v) -> str:
    if v is None:
        return ''
    if isinstance(v, (list, tuple)):
        return ', '.join(str(x) for x in v)
    return str(v)


def count_words(text: str) -> int:
    """Exact whitespace-delimited word count, without materializing the words.

    Counts transitions from whitespace to non-whitespace over the raw bytes. Measured on
    fanfic-shaped text this is 3.4x faster than len(text.split()) with IDENTICAL results
    -- 8.9 minutes versus 30.2 over 13M documents -- because split() allocates a list of
    every word only to take its length. At corpus scale that difference is most of the
    index build budget.
    """
    if not text:
        return 0
    b = np.frombuffer(text.encode('utf-8', 'ignore'), dtype=np.uint8)
    if b.size == 0:
        return 0
    ws = (b == 32) | (b == 10) | (b == 9) | (b == 13)
    return int(np.count_nonzero(~ws[1:] & ws[:-1])) + (0 if ws[0] else 1)


def flatten(rec: Dict, exclude: Optional[Set[str]] = None,
            derive: bool = True) -> Dict[str, str]:
    """Top-level scalars plus a flattened `metadata` dict, as strings.

    Everything is stored as text because that is how it arrives (AO3 writes `words` as
    "8"), and metaquery coerces on demand. One storage type keeps a single code path, and
    parquet dictionary-encodes the low-cardinality fields for free.

    Three derived fields are added so that size filtering works on EVERY record, not just
    the ~69% that declare a word count:

        words_actual    always present -- counted from the body at index time
        words_declared  the source value, empty where the corpus omitted it
        words           declared when present, otherwise the counted value
        words_ratio     actual / declared, blank when nothing was declared

    The declared value is preserved rather than overwritten, because the two mean
    different things: 'words_ratio' < 0.5 finds records whose text failed to extract,
    which is invisible once you have conflated "what the archive says" with "what we got".

    The body is read to count it and then discarded -- it is never stored in the index.
    """
    skip = EXCLUDE_FIELDS if exclude is None else (EXCLUDE_FIELDS | exclude)
    out: Dict[str, str] = {}
    body = ''
    for k, v in rec.items():
        if k in skip:
            if not body and isinstance(v, str):
                body = v
            continue
        if k == 'metadata' and isinstance(v, dict):
            # Legacy convention (AO3 et al.): children of a dict literally
            # named 'metadata' are indexed under BARE names. Preserved so
            # existing indexes and saved queries keep working.
            for mk, mv in v.items():
                if mk in skip:
                    if not body and isinstance(mv, str):
                        body = mv
                else:
                    out[mk] = _as_text(mv)
        elif isinstance(v, dict):
            # Any other dict column (RedPajama's `meta`, HF's `info`, ...):
            # flatten one level under DOTTED names -- the same addressing
            # convention as filter rules and tokenize templates (meta.title).
            for mk, mv in v.items():
                dk = f'{k}.{mk}'
                if dk in skip:
                    if not body and isinstance(mv, str):
                        body = mv
                elif not isinstance(mv, (dict, list)):
                    out[dk] = _as_text(mv)
        elif not isinstance(v, (dict, list)):
            out[k] = _as_text(v)

    if not derive:
        # Field DISCOVERY only needs key names, and the derived keys are constants. Word
        # counting during discovery would redo work the write pass repeats anyway -- on a
        # .zst tree that means decompressing and counting the first 200k records twice.
        out.update({k: '' for k in DERIVED_FIELDS})
        return out

    actual = count_words(body)
    declared = out.get('words', '').strip()
    out['words_declared'] = declared
    out['words_actual'] = str(actual)
    out['words'] = declared if declared else str(actual)
    try:
        d = int(declared.replace(',', ''))
        out['words_ratio'] = f"{actual / d:.4f}" if d > 0 else ''
    except ValueError:
        out['words_ratio'] = ''
    return out


def iter_records(paths: Sequence[Path], limit: Optional[int] = None):
    """Yield parsed records across files, in order.

    A malformed line still yields an (empty) record so that row numbering stays aligned
    with the explorer's global record indices -- silently dropping it would shift every
    subsequent index by one and quietly corrupt every query result.
    """
    n = 0
    for p in paths:
        with open_text(p) as fh:
            for line in fh:
                if limit is not None and n >= limit:
                    return
                n += 1
                try:
                    yield json.loads(line)
                except Exception:
                    yield {}


def discover_fields(paths: Sequence[Path], limit: int = DISCOVER_RECORDS,
                    exclude: Optional[Set[str]] = None, progress=None,
                    adaptive: bool = True):
    """Return (field_names, body_fields, records_scanned).

    body_fields maps a field to its mean value length for any field whose values look like
    prose rather than metadata. Detection is by LENGTH, not by name: a fixed list of names
    ('text', 'body', ...) silently fails on a corpus that calls its body 'story' or
    'chapter_text', and the failure mode is the entire corpus being written into the index.
    Length is self-configuring and does not care what the field is called.
    """
    seen: Dict[str, int] = {}
    total_len: Dict[str, int] = {}
    n = 0
    last_new = 0
    for rec in iter_records(paths, limit=limit):
        for k, v in flatten(rec, exclude=exclude, derive=False).items():
            if k not in seen:
                last_new = n + 1
            seen[k] = seen.get(k, 0) + 1
            total_len[k] = total_len.get(k, 0) + len(v)
        n += 1
        if adaptive and n >= DISCOVER_MIN and (n - last_new) >= DISCOVER_STABLE:
            break
        # Frequent relative to record cost: on a corpus of BOOKS each record is
        # megabytes, and this pass covers up to 200k of them -- silence here
        # reads as a hang.
        if progress is not None and n % 250 == 0:
            progress(n)
    fields = [k for k, _ in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))]
    body = {k: total_len[k] / seen[k] for k in seen
            if k not in DERIVED_FIELDS and seen[k]
            and total_len[k] / seen[k] > LONG_VALUE_CHARS}
    return fields, body, n


def build_index(paths: Sequence[Path], out_path: Path, progress=None,
                discover: int = DISCOVER_RECORDS,
                text_fields: Optional[Set[str]] = None,
                discover_progress=None) -> Dict:
    """Build the parquet index. Returns a stats dict; writes atomically via a .tmp file.

    `text_fields` names additional body fields to exclude, beyond the EXCLUDE_FIELDS
    defaults. Pass the caller's resolved text field: a corpus whose body lives under a name
    this module does not know ('story', 'chapter_text') would otherwise have its entire
    contents written into the index as a metadata column.
    """
    if not ARROW_AVAILABLE:
        raise RuntimeError("Building a metadata index needs pandas + pyarrow")

    hint = set(text_fields or ())
    fields, body_fields, discover_scanned = discover_fields(
        paths, discover, exclude=hint, progress=discover_progress)
    # Anything that looks like prose joins the exclusion set, and becomes the source the
    # word count is taken from. Detected before any row is written, so it never lands in
    # the index at all.
    skip = hint | set(body_fields)
    if body_fields:
        fields = [f for f in fields if f not in body_fields]
    if not fields:
        raise ValueError("No metadata fields found in the records scanned")

    schema = pa.schema([(f, pa.string()) for f in fields])
    tmp = out_path.with_name(out_path.name + '.tmp')
    buf: Dict[str, List[str]] = {f: [] for f in fields}
    known = set(fields)
    rows = 0
    unknown: Dict[str, int] = {}

    writer = pq.ParquetWriter(tmp, schema, compression='zstd')
    try:
        for rec in iter_records(paths):
            flat = flatten(rec, exclude=skip)
            for f in fields:
                buf[f].append(flat.get(f, ''))
            for k in flat:
                if k not in known:
                    unknown[k] = unknown.get(k, 0) + 1
            rows += 1
            # Progress decoupled from the write-flush cadence: CHUNK_ROWS (100k)
            # between callbacks is minutes of silence on large records.
            if progress is not None and rows % 250 == 0:
                progress(rows)
            if rows % CHUNK_ROWS == 0:
                writer.write_table(pa.table(buf, schema=schema))
                buf = {f: [] for f in fields}
        if buf[fields[0]]:
            writer.write_table(pa.table(buf, schema=schema))
    finally:
        writer.close()
    tmp.replace(out_path)
    if progress is not None:
        progress(rows)

    # Defence in depth: metadata values are short. A column averaging paragraphs of text
    # is almost certainly a body field this module failed to recognise, and it should not
    # be sitting in the index.
    return {'rows': rows, 'fields': fields, 'unknown_fields': unknown,
            'body_fields': {k: int(v) for k, v in body_fields.items()},
            'discover_scanned': discover_scanned,
            'size_mb': out_path.stat().st_size / (1024 * 1024)}


class MetadataIndex:
    """Read side: column-pruned access plus a small in-memory column cache.

    Satisfies the `table` protocol metaquery expects -- `.fields` and `.column(name)` -- so
    the same evaluator will work over a tokenized-shard sidecar later without change.
    """

    # Must exceed a typical field count, or a summary/query touching every column
    # evicts entries it is about to need again and re-reads them from disk.
    COLUMN_CACHE_MAX = 32

    def __init__(self, path: Path):
        if not ARROW_AVAILABLE:
            raise RuntimeError("Reading a metadata index needs pandas + pyarrow")
        self.path = Path(path)
        self._pf = pq.ParquetFile(self.path)
        self.fields: Set[str] = set(self._pf.schema_arrow.names)
        self.n_rows: int = self._pf.metadata.num_rows
        self._cache: Dict[str, "pd.Series"] = {}

    def resolve(self, name: str) -> str:
        """Field names match case-insensitively; the key itself must still exist."""
        if name in self.fields:
            return name
        lower = {f.lower(): f for f in self.fields}
        if name.lower() in lower:
            return lower[name.lower()]
        raise KeyError(name)

    def column(self, name: str) -> "pd.Series":
        real = self.resolve(name)
        hit = self._cache.get(real)
        if hit is None:
            # Only this column is read off disk -- the whole point of a columnar index.
            hit = self._pf.read(columns=[real]).column(real).to_pandas()
            if len(self._cache) >= self.COLUMN_CACHE_MAX:
                self._cache.pop(next(iter(self._cache)))
            self._cache[real] = hit
        return hit

    def field_summary(self, sample_rows: int = 300_000) -> List[Dict]:
        """Per-field fill rate and cardinality, estimated from a sample.

        Reports COUNTS, never example values: this index is built over corpora whose
        contents should not be splashed across a terminal by an inspection command. Use
        value_counts() to see values for one facet you name.

        SAMPLED ON PURPOSE. The exact version read every column in full and called
        nunique() on each -- hashing tens of millions of strings per field, holding
        gigabytes of Python objects, and thrashing the column cache. On a 13M-record index
        that took minutes with no output and looked like a hang. Row groups are sampled
        evenly across the file (not just the head, which would be biased by input order)
        and every column is read in ONE pass instead of one read per field.
        """
        n_rg = self._pf.num_row_groups
        if n_rg == 0 or self.n_rows == 0:
            return []
        rows_per_rg = max(1, self.n_rows // n_rg)
        want = max(1, min(n_rg, -(-sample_rows // rows_per_rg)))
        if want >= n_rg:
            picks = list(range(n_rg))
        elif want == 1:
            picks = [n_rg // 2]
        else:
            picks = sorted({int(round(i * (n_rg - 1) / (want - 1))) for i in range(want)})

        df = self._pf.read_row_groups(picks).to_pandas()
        sampled = len(df)
        exact = sampled >= self.n_rows

        out = []
        for f in sorted(self.fields):
            col = df[f].fillna('')
            nonempty = int((col.str.len() > 0).sum())
            out.append({'field': f,
                        'non_empty': int(round(nonempty / sampled * self.n_rows)),
                        'fill_pct': nonempty / sampled * 100,
                        'distinct': int(col.nunique()),
                        'derived': f in DERIVED_FIELD_DOC,
                        'sampled': sampled, 'exact': exact})
        return out

    def value_counts(self, field: str, top: int = 20) -> "pd.Series":
        """Most common values of ONE named field -- for choosing filters on low-cardinality
        facets like Language, Rating or Category. Explicit by design."""
        return self.column(self.resolve(field)).value_counts().head(top)
