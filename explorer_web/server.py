#!/usr/bin/env python3
"""
Dataset Explorer Web — FastAPI wrapper around dataset_explorer.DatasetExplorer.

Design:
  * One DatasetExplorer instance per loaded dataset, each owned by a single
    worker thread. All mutating / potentially-slow operations run as Jobs on
    that thread, which preserves the explorer's single-threaded assumptions.
  * Fast reads (record fetch via the line index, set listing) are served
    directly so the UI stays responsive while a long job (neardupe, export)
    runs. These touch disk + read-mostly state; worst case during a concurrent
    prune is a stale read, never corruption of the source data.
  * All print() output from the explorer is captured per-job via a
    thread-local stdout proxy and streamed to the browser over SSE. Rich/tqdm
    are disabled so logs are plain text.

Run:  python explorer_web/server.py [--host 0.0.0.0] [--port 8765]
"""

import argparse
import gzip
import hashlib
import io
import itertools
import json
import os
import pickle
import queue
import re
import string
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import dataset_explorer as dx

# Plain-text logs: Rich emits ANSI + control sequences and tqdm writes to stderr,
# neither of which survives the SSE log pane. The explorer's fallback prints
# periodic percentage lines, which is exactly what we want in a log stream.
dx.RICH_AVAILABLE = False
dx.TQDM_AVAILABLE = False

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Please install server dependencies:  pip install fastapi uvicorn")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Thread-local stdout: each worker thread routes print() into its current
# job's log; every other thread (uvicorn, main) falls through to the real one.
# ---------------------------------------------------------------------------

class ThreadLocalStdout(io.TextIOBase):
    def __init__(self, fallback):
        self._fallback = fallback
        self._local = threading.local()

    def set_target(self, target):
        self._local.target = target

    def clear_target(self):
        self._local.target = None

    def write(self, s):
        target = getattr(self._local, 'target', None)
        return (target or self._fallback).write(s)

    def flush(self):
        target = getattr(self._local, 'target', None)
        (target or self._fallback).flush()

    @property
    def encoding(self):
        return getattr(self._fallback, 'encoding', 'utf-8')


_stdout_proxy = ThreadLocalStdout(sys.stdout)
sys.stdout = _stdout_proxy


class JobLogWriter(io.TextIOBase):
    """Accumulates whole lines into job.log.

    Understands terminal-style '\r' progress rewrites: a line terminated by a lone
    '\r' is emitted immediately, and the NEXT line replaces it -- so a print(...,
    end='\r') progress loop appears as one updating log line instead of vanishing
    into the partial-line buffer forever.
    """

    def __init__(self, job: 'Job'):
        self.job = job
        self._buf = ''
        self._replace_next = False

    def write(self, s):
        self._buf = (self._buf + s).replace('\r\n', '\n')
        while True:
            i_n = self._buf.find('\n')
            i_r = self._buf.find('\r')
            # Hold a trailing '\r': it may be the first half of a '\r\n' split
            # across two writes.
            if i_r == len(self._buf) - 1 and i_n == -1:
                break
            if i_n == -1 and i_r == -1:
                break
            if i_r != -1 and (i_n == -1 or i_r < i_n):
                line, self._buf = self._buf[:i_r], self._buf[i_r + 1:]
                self._emit(line)
                self._replace_next = True
            else:
                line, self._buf = self._buf[:i_n], self._buf[i_n + 1:]
                self._emit(line)
                self._replace_next = False
        return len(s)

    def _emit(self, line: str):
        if self._replace_next and self.job.log:
            if line:                     # bare newline after a \r-line: keep the
                self.job.log[-1] = line  # final progress state, don't blank it
        else:
            self.job.log.append(line)

    def flush(self):
        pass


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

_job_counter = itertools.count(1)


class Job:
    def __init__(self, kind: str, dataset_id: Optional[str], params: Dict[str, Any]):
        self.id = f"job{next(_job_counter):04d}"
        self.kind = kind
        self.dataset_id = dataset_id
        self.params = params
        self.status = 'queued'          # queued | running | done | error | cancelled
        self.cancel_requested = False
        self.log: List[str] = []
        self.result: Any = None
        self.error: Optional[str] = None
        self.progress: Optional[Dict[str, Any]] = None
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None

    def summary(self, log_from: int = 0) -> Dict[str, Any]:
        return {
            'id': self.id, 'kind': self.kind, 'dataset_id': self.dataset_id,
            'status': self.status, 'error': self.error,
            'cancel_requested': self.cancel_requested,
            'created': self.created, 'started': self.started, 'finished': self.finished,
            'log_length': len(self.log),
            'log': self.log[log_from:],
            'progress': _jsonable(self.progress),
            'result': _jsonable(self.result) if self.status in ('done', 'error') else None,
            'params': {k: v for k, v in self.params.items() if k != 'fn'},
        }


JOBS: Dict[str, Job] = {}


class JobCancelled(BaseException):
    """Raised inside a job to unwind it on user cancellation.

    BaseException ON PURPOSE: report_progress guards its hook with
    'except Exception', and cancellation must escape that guard -- the same
    property the sketch-resume kill tests rely on.
    """


def _kill_proc_tree(proc):
    """Kill a subprocess AND its children (worker pools survive a plain kill)."""
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                           capture_output=True)
        else:
            import signal as _signal
            os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
    except Exception as e:
        print(f"(process-tree kill failed: {e})")


class CacheConflictError(Exception):
    """Cache adoption needs a human keep/discard decision before loading."""

    def __init__(self, conflicts):
        super().__init__(f"{len(conflicts)} cache conflict(s) need a keep/discard "
                         f"decision before this dataset can load safely")
        self.payload = {'conflicts': conflicts}

# The explorer's report_progress() calls land here, routed to the job currently
# executing on this thread. Two nesting levels: 'main' (e.g. file 12/400) and
# 'stage' (progress within the current step). ETA comes from each level's own
# elapsed/fraction ratio. Throttled to ~3 updates/sec per level.
_current_job = threading.local()


def _progress_hook(stage: str, done: int, total: int, main: bool,
                   note: Optional[str] = None, unit: Optional[str] = None):
    job = getattr(_current_job, 'job', None)
    if job is None:
        return
    if job.cancel_requested:
        # Cooperative cancellation: every long loop reports progress, so every
        # long loop is cancellable at its natural granularity. Checkpoint/
        # resume machinery makes this SAFE -- cancelled work resumes.
        raise JobCancelled()
    now = time.time()
    prog = job.progress or {'main': None, 'stage': None}
    slot = 'main' if main else 'stage'
    cur = prog.get(slot)
    if cur is None or cur.get('name') != stage:
        cur = {'name': stage, 't0': now}
    elif now - cur.get('t', 0) < 0.3 and (not total or done < total):
        return
    cur.update(done=done, total=total, t=now, note=note, unit=unit)
    if total:
        cur['pct'] = done / total * 100
        if done > 0:
            cur['eta_s'] = (now - cur['t0']) * (total - done) / done
    prog[slot] = cur
    prog['updated'] = now
    job.progress = prog


dx.PROGRESS_HOOK = _progress_hook


class DatasetWorker(threading.Thread):
    """Serializes all explorer operations for one dataset."""

    def __init__(self, dataset_id: str):
        super().__init__(daemon=True, name=f"worker-{dataset_id}")
        self.queue: 'queue.Queue[Optional[Job]]' = queue.Queue()
        self.dataset_id = dataset_id

    def submit(self, kind: str, fn, params: Optional[Dict[str, Any]] = None) -> Job:
        job = Job(kind, self.dataset_id, params or {})
        job.params['fn'] = fn
        JOBS[job.id] = job
        self.queue.put(job)
        return job

    def run(self):
        while True:
            job = self.queue.get()
            if job is None:
                return
            if job.cancel_requested:
                job.status = 'cancelled'
                job.finished = time.time()
                job.log.append('[cancelled before starting]')
                continue
            job.status = 'running'
            job.started = time.time()
            writer = JobLogWriter(job)
            _stdout_proxy.set_target(writer)
            _current_job.job = job
            try:
                job.result = job.params['fn']()
                job.status = 'done'
            except JobCancelled:
                job.status = 'cancelled'
                job.log.append('[cancelled by user]')
            except Exception as e:
                job.error = f"{type(e).__name__}: {e}"
                job.result = getattr(e, 'payload', None)   # structured error data
                if job.result is None:
                    job.log.extend(traceback.format_exc().splitlines())
                job.status = 'error'
            finally:
                if writer._buf:
                    writer.write('\n')
                _stdout_proxy.clear_target()
                _current_job.job = None
                job.finished = time.time()


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

_ds_counter = itertools.count(1)


class DatasetEntry:
    def __init__(self, path: str, opts: Dict[str, Any]):
        self.id = f"ds{next(_ds_counter):02d}"
        self.path = path
        self.opts = opts
        self.explorer: Optional[dx.DatasetExplorer] = None
        self.worker = DatasetWorker(self.id)
        self.worker.start()
        self.load_job: Optional[Job] = None

    @property
    def status(self) -> str:
        if self.explorer is not None:
            return 'ready'
        if self.load_job and self.load_job.status in ('error', 'cancelled'):
            return self.load_job.status
        return 'loading'

    def summary(self) -> Dict[str, Any]:
        out = {
            'id': self.id, 'path': self.path, 'status': self.status,
            'load_job': self.load_job.id if self.load_job else None,
            'opts': self.opts,
        }
        if self.explorer is not None:
            md = self.explorer.metadata
            out['info'] = _jsonable({
                'num_rows': md.get('num_rows'),
                'columns': md.get('columns'),
                'num_files': md.get('num_files', 1),
                'file_type': self.explorer.file_type,
                'file_size_mb': md.get('file_size'),
                'has_index': md.get('has_index'),
                'is_token_shard': md.get('is_token_shard', False),
            })
        return out


DATASETS: Dict[str, DatasetEntry] = {}

# Worker for jobs that belong to no dataset (cache migration, future maintenance).
UTILITY_WORKER = DatasetWorker('util')
UTILITY_WORKER.start()


# ---------------------------------------------------------------------------
# Managed-dataset registry: a persistent library of known datasets, above the
# ephemeral "load a path" layer. Pure metadata -- registering, editing, and
# unregistering NEVER touch the data files themselves.
# ---------------------------------------------------------------------------

class Registry:
    """JSON-file-backed dataset registry with atomic writes.

    Kept deliberately simple (one file, whole-file replace) so it can live on a
    NAS and be shared by every rig's server instance. Reloaded before each
    mutation so concurrent servers see each other's registrations; last writer
    wins on a true simultaneous edit, which is acceptable for a curation tool.
    """

    def __init__(self, path: Path):
        self.path = Path(path).expanduser()
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {'version': 1, 'datasets': {}}
        try:
            return json.loads(self.path.read_text(encoding='utf-8'))
        except Exception as e:
            raise HTTPException(500, f"Registry file unreadable ({self.path}): {e}")

    def _save(self, data: Dict[str, Any]):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(data, indent=1, default=str), encoding='utf-8')
        tmp.replace(self.path)

    def list(self) -> List[Dict[str, Any]]:
        data = self._load()
        return sorted(data['datasets'].values(),
                      key=lambda e: e.get('name', '').lower())

    def get(self, rid: str) -> Dict[str, Any]:
        entry = self._load()['datasets'].get(rid)
        if entry is None:
            raise HTTPException(404, f"No registered dataset {rid!r}")
        return entry

    def upsert(self, entry: Dict[str, Any]):
        with self._lock:
            data = self._load()
            data['datasets'][entry['id']] = entry
            self._save(data)

    def delete(self, rid: str):
        with self._lock:
            data = self._load()
            if rid not in data['datasets']:
                raise HTTPException(404, f"No registered dataset {rid!r}")
            del data['datasets'][rid]
            self._save(data)

    def unique_id(self, name: str) -> str:
        base = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-') or 'dataset'
        data = self._load()
        rid, i = base, 2
        while rid in data['datasets']:
            rid = f"{base}-{i}"
            i += 1
        return rid

    # ---- filters: corpus-AGNOSTIC named rule definitions ------------------
    # A filter is intensional (a definition that yields sets when evaluated
    # against a corpus) where a set is extensional (frozen indices). Filters
    # therefore live at library level, beside datasets, not under them.

    def list_filters(self) -> List[Dict[str, Any]]:
        data = self._load()
        return sorted(data.get('filters', {}).values(),
                      key=lambda f: f.get('name', '').lower())

    def get_filter(self, fid: str) -> Dict[str, Any]:
        f = self._load().get('filters', {}).get(fid)
        if f is None:
            raise HTTPException(404, f"No filter {fid!r}")
        return f

    def upsert_filter(self, entry: Dict[str, Any]):
        with self._lock:
            data = self._load()
            data.setdefault('filters', {})[entry['id']] = entry
            self._save(data)

    def delete_filter(self, fid: str):
        with self._lock:
            data = self._load()
            if fid not in data.get('filters', {}):
                raise HTTPException(404, f"No filter {fid!r}")
            del data['filters'][fid]
            data.get('filter_evals', {}).pop(fid, None)
            self._save(data)

    # ---- transforms: corpus-AGNOSTIC named rewrite chains -----------------
    # The fourth noun. A filter is a predicate (record -> bool, composes as
    # set algebra); a transform is a function (record -> record', composes by
    # SEQUENCING). They version independently: iterating a scrub chain must
    # not invalidate a filter's materialized drop sets, and one chain is
    # reusable under many drop policies.

    def list_transforms(self) -> List[Dict[str, Any]]:
        data = self._load()
        return sorted(data.get('transforms', {}).values(),
                      key=lambda t: t.get('name', '').lower())

    def get_transform(self, tid: str) -> Dict[str, Any]:
        t = self._load().get('transforms', {}).get(tid)
        if t is None:
            raise HTTPException(404, f"No transform {tid!r}")
        return t

    def upsert_transform(self, entry: Dict[str, Any]):
        with self._lock:
            data = self._load()
            data.setdefault('transforms', {})[entry['id']] = entry
            self._save(data)

    def delete_transform(self, tid: str):
        with self._lock:
            data = self._load()
            if tid not in data.get('transforms', {}):
                raise HTTPException(404, f"No transform {tid!r}")
            del data['transforms'][tid]
            data.get('transform_evals', {}).pop(tid, None)
            self._save(data)

    def migrate_filter_scrubs(self):
        """One-shot: scrubs born inside filters move out into transforms named
        `<filter>-scrubs`. Idempotent; filters keep rules only afterwards.
        Filter versions are NOT bumped (rules unchanged, r= stamps stay valid)."""
        with self._lock:
            data = self._load()
            moved = []
            for fid, f in data.get('filters', {}).items():
                if not f.get('scrubs'):
                    continue
                tname = f"{f['name']}-scrubs"
                tid = f"tfm-{re.sub(r'[^a-z0-9]+', '-', tname.lower()).strip('-')}"
                if tid not in data.setdefault('transforms', {}):
                    data['transforms'][tid] = {
                        'id': tid, 'name': tname,
                        'scrubs': f['scrubs'],
                        'fixpoint': bool(f.get('scrub_fixpoint')),
                        'notes': f"migrated from filter {f['name']!r}",
                        'version': 1,
                        'created': time.time(), 'updated': time.time(),
                    }
                f['scrubs'] = []
                f['scrub_fixpoint'] = False
                moved.append((f['name'], tname))
            if moved:
                self._save(data)
            return moved

    def record_transform_eval(self, tid: str, entry: Dict[str, Any]):
        with self._lock:
            data = self._load()
            evals = data.setdefault('transform_evals', {}).setdefault(tid, [])
            evals[:] = [e for e in evals
                        if not (e.get('path') == entry['path']
                                and e.get('version') == entry['version'])]
            evals.insert(0, entry)
            del evals[24:]
            self._save(data)

    def list_transform_evals(self, tid: str) -> List[Dict[str, Any]]:
        return self._load().get('transform_evals', {}).get(tid, [])

    # ---- filter eval history: (filter, corpus) observations ---------------
    # Persisted beside the filters so the A/B comparison (same filter, columns
    # per corpus) survives server restarts and is shared across rigs via NAS.

    def record_filter_eval(self, fid: str, entry: Dict[str, Any]):
        with self._lock:
            data = self._load()
            evals = data.setdefault('filter_evals', {}).setdefault(fid, [])
            # same corpus+version+mode re-run replaces its old column
            evals[:] = [e for e in evals
                        if not (e.get('path') == entry['path']
                                and e.get('version') == entry['version']
                                and e.get('sample') == entry['sample'])]
            evals.insert(0, entry)
            del evals[24:]
            self._save(data)

    def list_filter_evals(self, fid: str) -> List[Dict[str, Any]]:
        return self._load().get('filter_evals', {}).get(fid, [])

    def clear_filter_evals(self, fid: str):
        with self._lock:
            data = self._load()
            data.get('filter_evals', {}).pop(fid, None)
            self._save(data)

    def find_by_path(self, path: Path) -> Optional[Dict[str, Any]]:
        target = str(Path(path).resolve())
        for e in self._load()['datasets'].values():
            try:
                if str(Path(e['path']).resolve()) == target:
                    return e
            except OSError:
                continue
        return None


REGISTRY: Optional[Registry] = None    # set in main() from --registry
TOKENIZED_ROOT: Optional[str] = None   # set in main() from --tokenized-root


def _registry() -> Registry:
    if REGISTRY is None:
        raise HTTPException(503, "Registry not configured")
    return REGISTRY


SUPPORTED_SUFFIXES = ('.parquet', '.jsonl', '.json', '.npy')


def _is_data_file(p: Path) -> bool:
    return p.is_file() and (p.suffix.lower() in SUPPORTED_SUFFIXES
                            or p.name.lower().endswith('.jsonl.zst'))


def _cached_num_rows(path: Path) -> Optional[int]:
    """Record count from the explorer's own metadata cache, if one exists.
    Read-only: never creates the cache directory (the path may be a NAS mount
    we shouldn't write to just for registering)."""
    try:
        cache_dir = path.parent / '.dataset_explorer_cache'
        if not cache_dir.is_dir():
            return None
        file_hash = hashlib.md5(str(path.absolute()).encode()).hexdigest()[:8]
        stem = path.stem
        if stem.endswith('.jsonl'):
            stem = stem[:-6]
        meta = cache_dir / f"{stem}_{file_hash}.meta.gz"
        if not meta.exists():
            return None
        with gzip.open(meta, 'rb') as f:
            return pickle.load(f).get('metadata', {}).get('num_rows')
    except Exception:
        return None


def _manifest_stats(path: Path) -> Dict[str, Optional[int]]:
    """Token/doc counts from pre_tokenize manifests. Cheap: one glob + a few
    small file reads, regardless of shard count."""
    tokens = docs = 0
    if path.is_dir():
        for m in path.glob('manifest_*.json'):
            try:
                d = json.loads(m.read_text(encoding='utf-8'))
                tokens += int(d.get('tokens') or 0)
                docs += int(d.get('docs') or 0)
            except Exception:
                continue
    return {'tokens': tokens or None, 'docs': docs or None}


# Beyond this many files, _quick_stats stops stat()ing for exact sizes -- a
# synchronous registration once spent a minute doing 5,843 SMB stats for a
# number the UI shows as '—' anyway.
QUICK_STATS_STAT_CAP = 1000


def _quick_stats(path: Path) -> Dict[str, Any]:
    """Cheap vitals for a registry entry: sizes from stat() (BOUNDED), token/
    doc counts from pre_tokenize manifests when present, record counts from
    the explorer's cache when one exists. Never scans file contents."""
    stats: Dict[str, Any] = {'size_mb': None, 'num_rows': None, 'tokens': None,
                             'num_files': None, 'kind_guess': 'text'}
    if path.is_dir():
        files = [p for p in path.iterdir() if _is_data_file(p)]
        npys = [p for p in files if p.suffix.lower() == '.npy']
        stats['num_files'] = len(npys) if npys else len(files)
        if len(files) <= QUICK_STATS_STAT_CAP:
            stats['size_mb'] = sum(p.stat().st_size for p in files) / 1e6
        if npys:
            stats['kind_guess'] = 'tokenized'
        ms = _manifest_stats(path)
        if ms['tokens']:
            stats['tokens'] = ms['tokens']
        if ms['docs']:
            stats['num_rows'] = ms['docs']
    else:
        stats['num_files'] = 1
        stats['size_mb'] = path.stat().st_size / 1e6
        if path.suffix.lower() == '.npy':
            stats['kind_guess'] = 'tokenized'
        if stats['num_rows'] is None:
            stats['num_rows'] = _cached_num_rows(path)
    return stats


def _entry(dataset_id: str) -> DatasetEntry:
    entry = DATASETS.get(dataset_id)
    if entry is None:
        raise HTTPException(404, f"No dataset {dataset_id!r}")
    return entry


def _explorer(dataset_id: str) -> dx.DatasetExplorer:
    entry = _entry(dataset_id)
    if entry.explorer is None:
        raise HTTPException(409, f"Dataset {dataset_id!r} is not ready (status: {entry.status})")
    return entry.explorer


def _jsonable(obj):
    """Recursively convert numpy / pandas scalars, arrays and Paths for JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _get_record_dict(explorer: dx.DatasetExplorer, index: int) -> Dict[str, Any]:
    rec = None
    if explorer.file_type == 'jsonl':
        rec = explorer.get_record_by_position(index)
    if rec is None:
        df = explorer.get_record(index)
        rec = df.iloc[0].to_dict()
    return _jsonable(rec)


def _set_indices(explorer: dx.DatasetExplorer, name: str) -> np.ndarray:
    entry = explorer.result_sets.get(name)
    if entry is None:
        raise HTTPException(404, f"No result set named {name!r}")
    return np.asarray(entry['indices'])


def _truncate(value: Any, width: int) -> Any:
    s = value if isinstance(value, str) else json.dumps(_jsonable(value), ensure_ascii=False)
    return s[:width] + ('…' if len(s) > width else '')


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class LoadRequest(BaseModel):
    path: str
    quick: bool = False
    no_cache: bool = False
    rebuild_cache: bool = False
    raw_shards: bool = False
    recursive: bool = False
    text_field: Optional[str] = None
    tok_kind: Optional[str] = None
    tok_path: Optional[str] = None
    special_tokens: Optional[str] = None
    npy_max_docs: Optional[int] = None


class RegisterRequest(BaseModel):
    path: str
    name: str
    tags: List[str] = []
    notes: str = ''
    kind: Optional[str] = None          # text | tokenized (None = auto-guess)
    derived_from: Optional[str] = None  # registry id of the parent dataset
    # remembered open options
    text_field: Optional[str] = None
    tok_kind: Optional[str] = None
    tok_path: Optional[str] = None
    special_tokens: Optional[str] = None
    raw_shards: bool = False
    recursive: bool = False


class RegistryEditRequest(BaseModel):
    name: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    kind: Optional[str] = None
    derived_from: Optional[str] = None
    text_field: Optional[str] = None
    tok_kind: Optional[str] = None
    tok_path: Optional[str] = None
    special_tokens: Optional[str] = None
    raw_shards: Optional[bool] = None
    recursive: Optional[bool] = None


class SearchRequest(BaseModel):
    mode: str = 'text'              # text | meta
    query: str = ''
    terms: Optional[List[str]] = None   # multi-term AND match (text mode)
    field: Optional[str] = None
    regex: bool = False
    limit: Optional[int] = None
    name: Optional[str] = None
    within_set: Optional[str] = None    # intersect results with this set


class SetOpRequest(BaseModel):
    op: str                          # union | intersect | subtract
    sets: List[str]
    name: Optional[str] = None


class RenameRequest(BaseModel):
    old: str
    new: str


class ExportRequest(BaseModel):
    out_path: str
    include: Optional[List[str]] = None   # union of these sets (default: all records)
    exclude: Optional[List[str]] = None   # minus union of these
    transform_id: Optional[str] = None    # rewrite chain applied while writing
    mode: str = 'single'                  # single | shards (mirror) | split (size-capped)
    split_mb: float = 300.0               # split mode: roll to a new file at this size
    register_as: Optional[str] = None     # library name for the result (with lineage)


class ExportPlanRequest(BaseModel):
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None


class StatsRequest(BaseModel):
    field: Optional[str] = None


class NeardupeRequest(BaseModel):
    threshold: float = 0.8
    ngram: int = 13
    perms: int = 1024
    field: Optional[str] = None
    sample: Optional[int] = None
    device: str = 'cuda'
    exact_cardinality: bool = False
    rebuild: bool = False
    min_tokens: int = 0
    set_name: Optional[str] = None    # base name for the <name>/<name>_cut sets


class PruneRequest(BaseModel):
    out_dir: str
    write: bool = False
    keep: str = 'longest'
    include_chains: bool = False
    include_short: bool = False
    protect_val: bool = True


class MetaValuesRequest(BaseModel):
    field: str
    top: int = 20


# ---------------------------------------------------------------------------
# App + endpoints
# ---------------------------------------------------------------------------

app = FastAPI(title="Dataset Explorer Web")
STATIC_DIR = Path(__file__).resolve().parent / 'static'


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / 'index.html')


@app.get("/api/datasets")
def list_datasets():
    return [e.summary() for e in DATASETS.values()]


def _start_load(req: LoadRequest) -> DatasetEntry:
    path = Path(req.path).expanduser()
    if not path.exists():
        raise HTTPException(400, f"Path does not exist: {path}")
    entry = DatasetEntry(str(path), req.model_dump())
    DATASETS[entry.id] = entry

    def _load():
        try:
            return _load_inner()
        except BaseException:
            # Cancelled or failed load: remove the placeholder entry so the
            # sidebar doesn't keep a permanent 'loading' ghost. The job (and
            # its log) remain in the Jobs tab for diagnosis, and a registered
            # dataset's row simply reverts to its 'open' button.
            DATASETS.pop(entry.id, None)
            entry.worker.queue.put(None)      # let the worker thread exit
            raise

    def _load_inner():
        # Caches keyed for another spelling of this path (moved dataset, mapped
        # drive vs UNC, pre-marker copy)? Adopt them FIRST -- rename-only,
        # idempotent, content re-validated on load -- so nothing re-derives.
        # If adoption hits a conflict only a human can settle, STOP before any
        # derivation: the UI turns the structured conflicts into a chooser and
        # retries the load.
        if dx.needs_adoption(path):
            print(f"Detected caches keyed for a different path spelling; "
                  f"adopting for {path} before load...")
            migrate_report = dx.migrate_cache(str(path))
            if migrate_report['ambiguous']:
                raise CacheConflictError(migrate_report['ambiguous'])
        explorer = dx.DatasetExplorer(
            str(path), quick_mode=req.quick, no_cache=req.no_cache,
            rebuild_cache=req.rebuild_cache, tok_kind=req.tok_kind,
            tok_path=req.tok_path, special_tokens=req.special_tokens,
            npy_max_docs=req.npy_max_docs, dedup_only=req.raw_shards,
            text_field=req.text_field, non_interactive=True,
            recursive=req.recursive,
        )
        entry.explorer = explorer
        # If this path is registered, write fresh vitals back to the library.
        if REGISTRY is not None:
            try:
                reg = REGISTRY.find_by_path(path)
                if reg is not None:
                    md = explorer.metadata
                    reg.setdefault('stats', {}).update(_jsonable({
                        'num_rows': md.get('num_rows'),
                        'size_mb': md.get('file_size'),
                        'num_files': md.get('num_files', 1),
                    }))
                    reg['last_opened'] = time.time()
                    REGISTRY.upsert(reg)
            except Exception as e:
                print(f"(registry stats update skipped: {e})")
        return {'num_rows': explorer.metadata.get('num_rows')}

    entry.load_job = entry.worker.submit('load', _load, {'path': str(path)})
    return entry


@app.post("/api/datasets")
def load_dataset(req: LoadRequest):
    entry = _start_load(req)
    return {'dataset_id': entry.id, 'job_id': entry.load_job.id}


@app.get("/api/datasets/{dataset_id}")
def dataset_info(dataset_id: str):
    entry = _entry(dataset_id)
    out = entry.summary()
    if entry.explorer is not None:
        out['metadata'] = _jsonable(entry.explorer.metadata)
        if entry.explorer.is_directory:
            out['files'] = [
                {'name': p.name, 'records': int(c)}
                for p, c in zip(entry.explorer.source_files,
                                entry.explorer.file_record_counts)
            ]
        # One level of nested keys from record 0, so dict-typed columns
        # (meta: {...}) aren't opaque on the Info page. Representative, not
        # exhaustive -- the metaindex field summary is the exhaustive tool.
        try:
            rec0 = _get_record_dict(entry.explorer, 0)
            nested = {k: sorted(v.keys()) for k, v in rec0.items()
                      if isinstance(v, dict) and v}
            if nested:
                out['nested_fields'] = nested
        except Exception:
            pass
    return out


@app.delete("/api/datasets/{dataset_id}")
def close_dataset(dataset_id: str):
    entry = _entry(dataset_id)
    entry.worker.queue.put(None)     # stop worker after any in-flight job
    del DATASETS[dataset_id]
    return {'closed': dataset_id}


# ---- records --------------------------------------------------------------

@app.get("/api/datasets/{dataset_id}/record/{index}")
def get_record(dataset_id: str, index: int):
    explorer = _explorer(dataset_id)
    try:
        return {'index': index, 'record': _get_record_dict(explorer, index)}
    except (IndexError, ValueError) as e:
        raise HTTPException(404, str(e))


@app.get("/api/datasets/{dataset_id}/records")
def list_records(dataset_id: str, start: int = 0, limit: int = 50,
                 set_name: Optional[str] = None, preview: int = 200):
    """A page of record previews, over the whole dataset or one result set."""
    explorer = _explorer(dataset_id)
    limit = max(1, min(limit, 500))
    if set_name:
        indices = _set_indices(explorer, set_name)
        total = int(indices.size)
        page = [int(i) for i in indices[start:start + limit]]
    else:
        total = int(explorer.metadata.get('num_rows') or 0)
        page = list(range(start, min(start + limit, total)))

    rows = []
    for gi in page:
        try:
            rec = _get_record_dict(explorer, gi)
            rows.append({'index': gi,
                         'fields': {k: _truncate(v, preview) for k, v in rec.items()}})
        except Exception as e:
            rows.append({'index': gi, 'error': str(e)})
    return {'total': total, 'start': start, 'records': rows}


# ---- search / sets --------------------------------------------------------

@app.post("/api/datasets/{dataset_id}/search")
def search(dataset_id: str, req: SearchRequest):
    entry = _entry(dataset_id)
    explorer = _explorer(dataset_id)

    within = None
    if req.within_set:
        within = _set_indices(explorer, req.within_set)   # validate before queueing

    def _search():
        if req.mode == 'meta':
            indices = explorer.find_records_by_metadata(req.query, limit=req.limit)
            kind, qtext = 'meta', req.query
        else:
            terms = req.terms if req.terms else [req.query]
            terms = [t for t in terms if t]
            if not terms:
                raise ValueError("Empty query")
            field = req.field
            if field is not None:
                resolved = explorer._resolve_field(field)
                if resolved is None:
                    raise ValueError(f"Field {field!r} not found. Available: "
                                     f"{', '.join(explorer.metadata.get('columns') or [])}")
                field = resolved
            query_arg = terms if len(terms) > 1 else terms[0]
            indices = explorer.find_all_records(query_arg, field=field,
                                                regex=req.regex, limit=req.limit)
            kind = 'token' if explorer.file_type == 'npy' else 'text'
            qtext = str(query_arg)
        if within is not None:
            before = len(indices)
            indices = np.intersect1d(np.asarray(indices, dtype=np.int64),
                                     within.astype(np.int64)).tolist()
            print(f"Intersected with set {req.within_set!r}: {before:,} -> {len(indices):,}")
            qtext = f"{qtext} ∩ {req.within_set}"
        if not indices:
            return {'count': 0, 'set': None}
        name = explorer.store_result_set(indices, qtext, kind,
                                         name=req.name, activate=False)
        return {'count': len(indices), 'set': name,
                'limit_hit': req.limit is not None and len(indices) >= req.limit}

    job = entry.worker.submit('search', _search, req.model_dump())
    return {'job_id': job.id}


@app.get("/api/datasets/{dataset_id}/sets")
def list_sets(dataset_id: str):
    explorer = _explorer(dataset_id)
    total = explorer.metadata.get('num_rows') or 0
    return {
        'total_records': total,
        'sets': [
            {'name': name, 'count': e['count'], 'kind': e['kind'],
             'query': e['query'], 'created': e['created'],
             'pct': e['count'] / max(total, 1) * 100}
            for name, e in sorted(explorer.result_sets.items(),
                                  key=lambda kv: -kv[1]['created'])
        ],
    }


@app.post("/api/datasets/{dataset_id}/sets/ops")
def set_ops(dataset_id: str, req: SetOpRequest):
    explorer = _explorer(dataset_id)
    if req.op not in ('union', 'intersect', 'subtract'):
        raise HTTPException(400, f"Unknown op {req.op!r}")
    if len(req.sets) < 2:
        raise HTTPException(400, "Need at least two sets")
    arrays = [_set_indices(explorer, n).astype(np.int64) for n in req.sets]
    acc = arrays[0]
    for arr in arrays[1:]:
        if req.op == 'union':
            acc = np.union1d(acc, arr)
        elif req.op == 'intersect':
            acc = np.intersect1d(acc, arr)
        else:
            acc = np.setdiff1d(acc, arr)
    sym = {'union': ' ∪ ', 'intersect': ' ∩ ', 'subtract': ' − '}[req.op]
    name = explorer.store_result_set(acc.tolist(), sym.join(req.sets), 'combine',
                                     name=req.name, activate=False)
    return {'set': name, 'count': int(acc.size)}


@app.post("/api/datasets/{dataset_id}/sets/rename")
def rename_set(dataset_id: str, req: RenameRequest):
    explorer = _explorer(dataset_id)
    if req.old not in explorer.result_sets:
        raise HTTPException(404, f"No result set named {req.old!r}")
    if req.new in explorer.result_sets:
        raise HTTPException(409, f"A set named {req.new!r} already exists")
    explorer.result_sets[req.new] = explorer.result_sets.pop(req.old)
    explorer._save_sets()
    return {'renamed': [req.old, req.new]}


@app.delete("/api/datasets/{dataset_id}/sets/{name}")
def delete_set(dataset_id: str, name: str):
    explorer = _explorer(dataset_id)
    if not explorer.delete_result_set(name):
        raise HTTPException(404, f"No result set named {name!r}")
    return {'deleted': name}


# ---- export ---------------------------------------------------------------

def _export_keep(explorer, include, exclude) -> np.ndarray:
    total = int(explorer.metadata.get('num_rows') or 0)
    if include:
        keep = np.zeros(0, dtype=np.int64)
        for n in include:
            keep = np.union1d(keep, _set_indices(explorer, n).astype(np.int64))
    else:
        keep = np.arange(total, dtype=np.int64)
    for n in (exclude or []):
        keep = np.setdiff1d(keep, _set_indices(explorer, n).astype(np.int64))
    return keep


def _kept_bytes(positions, file_size: int, sel_local: np.ndarray) -> Optional[int]:
    """Exact byte total of the selected records, from the line index."""
    if positions is None or sel_local.size == 0:
        return 0 if sel_local.size == 0 else None
    pos = np.asarray(positions, dtype=np.int64)
    sizes = np.diff(np.append(pos, np.int64(file_size)))
    return int(sizes[sel_local].sum())


def _iter_kept_lines(explorer, keep: np.ndarray):
    """Raw JSONL line bytes for every kept record, in global order."""
    if explorer.file_type == 'jsonl' and explorer.is_directory:
        cum = explorer.cum_record_counts
        for fi, wpath in enumerate(explorer.working_files):
            lo, hi = cum[fi], cum[fi + 1]
            sel = keep[np.searchsorted(keep, lo):np.searchsorted(keep, hi)] - lo
            if sel.size == 0:
                continue
            dx.report_progress('export files', fi,
                               len(explorer.working_files), main=True)
            positions = explorer.file_line_positions[fi]
            with open(wpath, 'rb') as fin:
                if positions is not None:
                    for li in sel:
                        fin.seek(positions[int(li)])
                        yield fin.readline()
                else:
                    sset = set(int(x) for x in sel)
                    for ln, line in enumerate(fin):
                        if ln in sset:
                            yield line
    elif explorer.file_type == 'jsonl' and explorer.line_positions:
        with open(explorer.filepath, 'rb') as fin:
            for gi in keep:
                fin.seek(explorer.line_positions[int(gi)])
                yield fin.readline()
    else:
        for gi in keep:
            rec = _get_record_dict(explorer, int(gi))
            yield (json.dumps(rec, ensure_ascii=False) + '\n').encode()


@app.post("/api/datasets/{dataset_id}/export/plan")
def export_plan(dataset_id: str, req: ExportPlanRequest):
    """What an export recipe WOULD produce: exact record counts and (for
    indexed JSONL) exact byte sizes, per source file. Read-only and fast --
    pure set algebra plus line-index arithmetic, no data scan."""
    explorer = _explorer(dataset_id)
    total = int(explorer.metadata.get('num_rows') or 0)
    keep = _export_keep(explorer, req.include, req.exclude)

    files = []
    exact = True
    if explorer.is_directory:
        cum = explorer.cum_record_counts
        for fi, src in enumerate(explorer.source_files):
            lo, hi = cum[fi], cum[fi + 1]
            k_lo = int(np.searchsorted(keep, lo))
            k_hi = int(np.searchsorted(keep, hi))
            sel_local = keep[k_lo:k_hi] - lo
            wpath = explorer.working_files[fi]
            b = (_kept_bytes(explorer.file_line_positions[fi],
                             wpath.stat().st_size, sel_local)
                 if explorer.file_type == 'jsonl' else None)
            if b is None:
                exact = False
            files.append({'name': src.name, 'records': int(hi - lo),
                          'kept': int(sel_local.size), 'bytes': b})
    else:
        b = (_kept_bytes(explorer.line_positions,
                         explorer.filepath.stat().st_size, keep)
             if explorer.file_type == 'jsonl' and explorer.line_positions else None)
        if b is None:
            exact = False
        files.append({'name': explorer.original_filepath.name,
                      'records': total, 'kept': int(keep.size), 'bytes': b})

    known = [f['bytes'] for f in files if f['bytes'] is not None]
    return {
        'total_records': total,
        'kept_records': int(keep.size),
        'dropped_records': total - int(keep.size),
        'pct': keep.size / max(total, 1) * 100,
        'bytes': int(sum(known)) if known else None,
        'bytes_exact': exact and bool(known),
        'files': files,
    }

def _register_export(explorer, out_path: Path, req: 'ExportRequest', n_written: int):
    """Register an export result in the library with lineage back to its source."""
    if REGISTRY is None or not req.register_as:
        return
    try:
        if REGISTRY.find_by_path(out_path):
            print(f"  (already registered: {out_path})")
            return
        parent = REGISTRY.find_by_path(explorer.original_filepath)
        recipe = (f"export of {explorer.original_filepath.name}: "
                  f"include={req.include or 'ALL'} exclude={req.exclude or []}"
                  + (f" transform={req.transform_id}"
                     if getattr(req, 'transform_id', None) else ''))
        oo = {'text_field': explorer.text_field, 'tok_kind': None,
              'tok_path': None, 'special_tokens': None, 'raw_shards': False}
        stats = _quick_stats(out_path)
        stats.pop('kind_guess', None)
        stats['num_rows'] = n_written
        REGISTRY.upsert({
            'id': REGISTRY.unique_id(req.register_as),
            'name': req.register_as,
            'path': str(out_path.absolute()),
            'kind': 'text',
            'tags': ['export'],
            'notes': recipe,
            'derived_from': parent['id'] if parent else None,
            'open_opts': oo,
            'stats': _jsonable(stats),
            'created': time.time(),
            'last_opened': None,
        })
        print(f"  registered in library as {req.register_as!r}"
              + (f" (derived from {parent['name']})" if parent else ""))
    except Exception as e:
        print(f"  (library registration failed: {e})")


@app.post("/api/datasets/{dataset_id}/export")
def export_sets(dataset_id: str, req: ExportRequest):
    entry = _entry(dataset_id)
    explorer = _explorer(dataset_id)
    out_path = Path(req.out_path).expanduser()
    if req.mode not in ('single', 'shards', 'split'):
        raise HTTPException(400, f"Unknown export mode {req.mode!r}")
    if req.mode == 'single' and out_path.exists():
        raise HTTPException(409, f"Refusing to overwrite existing file: {out_path}")
    if req.mode == 'split' and req.split_mb <= 0:
        raise HTTPException(400, "split_mb must be positive")
    if req.mode in ('shards', 'split'):
        if req.mode == 'shards' and (not explorer.is_directory
                                     or explorer.file_type != 'jsonl'):
            raise HTTPException(400, "Shard-mirroring export needs a directory "
                                     "JSONL dataset; use single or split mode.")
        if out_path.exists() and any(out_path.iterdir()):
            raise HTTPException(409, f"Output directory not empty: {out_path}")
        try:
            if explorer.original_filepath.is_dir() \
                    and out_path.resolve() == explorer.original_filepath.resolve():
                raise HTTPException(400, "Output must differ from the source directory.")
        except OSError:
            pass

    for name in (req.include or []) + (req.exclude or []):
        _set_indices(explorer, name)     # validate before queueing

    tdef = _registry().get_transform(req.transform_id) if req.transform_id else None
    scrubs = (_compile_scrubs(tdef['scrubs'], explorer._resolve_text_field(None))
              if tdef else [])
    scrub_fixpt = bool(tdef.get('fixpoint')) if tdef else False
    scrub_counts: Dict[str, List[int]] = {}

    def _xform(line):
        """Apply the export's transform chain to one raw JSONL line (bytes or
        str, returned in kind). Identity when no transform is selected."""
        if not scrubs:
            return line
        rec = json.loads(line)
        _apply_scrubs(rec, scrubs, scrub_counts, fixpoint=scrub_fixpt)
        out = json.dumps(rec, ensure_ascii=False) + '\n'
        return out.encode() if isinstance(line, (bytes, bytearray)) else out

    def _print_scrub_totals():
        for sname, c in scrub_counts.items():
            print(f"  [scrub] {sname}: docs={c[0]:,} subs={c[1]:,} "
                  f"chars_removed={c[2]:,}")

    def _export_shards():
        keep = _export_keep(explorer, req.include, req.exclude)
        total = int(explorer.metadata.get('num_rows') or 0)
        out_path.mkdir(parents=True, exist_ok=True)
        cum = explorer.cum_record_counts
        n_written = 0
        n_files = len(explorer.source_files)
        print(f"Exporting {keep.size:,} of {total:,} records -> {out_path} "
              f"(mirroring {n_files} source shards)")
        for fi, src in enumerate(explorer.source_files):
            dx.report_progress('export shards', fi, n_files, main=True)
            lo, hi = cum[fi], cum[fi + 1]
            sel = keep[np.searchsorted(keep, lo):np.searchsorted(keep, hi)] - lo
            if sel.size == 0:
                print(f"  {src.name}: 0 kept, skipped")
                continue
            try:                 # preserve subdirectory structure (recursive datasets)
                rel = src.relative_to(explorer.original_filepath)
            except ValueError:
                rel = Path(src.name)
            name = rel.name[:-4] if rel.name.lower().endswith('.zst') else rel.name
            wpath = explorer.working_files[fi]
            positions = explorer.file_line_positions[fi]
            dest = out_path / rel.parent / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + '.part')
            with open(wpath, 'rb') as fin, open(tmp, 'wb') as fout:
                if positions is not None:
                    for li in sel:
                        fin.seek(positions[int(li)])
                        fout.write(_xform(fin.readline()))
                        n_written += 1
                        if n_written % 2000 == 0:
                            dx.report_progress('export', n_written, keep.size)
                else:
                    sel_set = set(int(x) for x in sel)
                    for ln, line in enumerate(fin):
                        if ln in sel_set:
                            fout.write(_xform(line))
                            n_written += 1
            tmp.replace(dest)
            print(f"  {name}: kept {sel.size:,} of {hi - lo:,}")
        print(f"Done: {n_written:,} records -> {out_path}")
        _print_scrub_totals()
        _register_export(explorer, out_path, req, n_written)
        return {'written': n_written, 'path': str(out_path), 'mode': 'shards'}

    def _export_split():
        keep = _export_keep(explorer, req.include, req.exclude)
        total = int(explorer.metadata.get('num_rows') or 0)
        limit = int(req.split_mb * 1e6)
        out_path.mkdir(parents=True, exist_ok=True)
        base = explorer.original_filepath
        stem = base.name if base.is_dir() else base.stem
        if stem.endswith('.jsonl'):          # x.jsonl.zst -> stem 'x.jsonl'
            stem = stem[:-6]
        stem = re.sub(r'[^A-Za-z0-9._-]+', '_', stem) or 'part'
        print(f"Exporting {keep.size:,} of {total:,} records -> {out_path} "
              f"(splitting at {req.split_mb:g} MB as {stem}_NNNNN.jsonl)")

        n_written = 0
        part_idx = 0
        cur = None
        cur_bytes = 0
        cur_tmp = cur_dest = None

        def roll():
            nonlocal cur, cur_tmp, cur_dest, cur_bytes, part_idx
            if cur is not None:
                cur.close()
                cur_tmp.replace(cur_dest)
                print(f"  {cur_dest.name}: {cur_bytes / 1e6:.1f} MB")
            cur_dest = out_path / f"{stem}_{part_idx:05d}.jsonl"
            cur_tmp = cur_dest.with_name(cur_dest.name + '.part')
            cur = open(cur_tmp, 'wb')
            cur_bytes = 0
            part_idx += 1

        roll()
        for line in _iter_kept_lines(explorer, keep):
            line = _xform(line)
            if not line.endswith(b'\n'):
                line += b'\n'
            if cur_bytes and cur_bytes + len(line) > limit:
                roll()
            cur.write(line)
            cur_bytes += len(line)
            n_written += 1
            if n_written % 2000 == 0:
                dx.report_progress('export', n_written, keep.size,
                                   note=f"part {part_idx}")
        cur.close()
        cur_tmp.replace(cur_dest)
        print(f"  {cur_dest.name}: {cur_bytes / 1e6:.1f} MB")
        print(f"Done: {n_written:,} records -> {part_idx} file(s) in {out_path}")
        _print_scrub_totals()
        _register_export(explorer, out_path, req, n_written)
        return {'written': n_written, 'path': str(out_path),
                'mode': 'split', 'parts': part_idx}

    def _export():
        keep = _export_keep(explorer, req.include, req.exclude)
        total = int(explorer.metadata.get('num_rows') or 0)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_written = 0
        print(f"Exporting {keep.size:,} of {total:,} records -> {out_path}")
        part = out_path.with_name(out_path.name + '.part')

        if explorer.file_type == 'jsonl' and not explorer.is_directory \
                and explorer.line_positions:
            # Fast path: seek straight to each record's byte offset.
            with open(explorer.filepath, 'rb') as fin, open(part, 'wb') as fout:
                for gi in keep:
                    fin.seek(explorer.line_positions[int(gi)])
                    fout.write(_xform(fin.readline()))
                    n_written += 1
                    if n_written % 1000 == 0:
                        dx.report_progress('export', n_written, keep.size)
                    if n_written % 100_000 == 0:
                        print(f"  {n_written:,} / {keep.size:,} written...")
        else:
            keep_set = set(int(i) for i in keep)
            with open(part, 'w', encoding='utf-8') as fout:
                if explorer.file_type == 'jsonl' and explorer.is_directory:
                    gi = 0
                    for fi, fpath in enumerate(explorer.working_files):
                        dx.report_progress('export files', fi,
                                           len(explorer.working_files), main=True)
                        with open(fpath, 'r', encoding='utf-8') as fin:
                            for line in fin:
                                if gi in keep_set:
                                    line = _xform(line)
                                    fout.write(line if line.endswith('\n') else line + '\n')
                                    n_written += 1
                                    if n_written % 1000 == 0:
                                        dx.report_progress('export', n_written, keep.size)
                                gi += 1
                else:
                    # parquet / npy: per-record fetch (decodes on demand for shards)
                    for gi in sorted(keep_set):
                        rec = _get_record_dict(explorer, gi)
                        if scrubs:
                            _apply_scrubs(rec, scrubs, scrub_counts,
                                          fixpoint=scrub_fixpt)
                        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                        n_written += 1
                        if n_written % 100 == 0:
                            dx.report_progress('export', n_written, keep.size)
                        if n_written % 10_000 == 0:
                            print(f"  {n_written:,} / {keep.size:,} written...")
        part.replace(out_path)
        print(f"Done: {n_written:,} records -> {out_path}")
        _print_scrub_totals()
        _register_export(explorer, out_path, req, n_written)
        return {'written': n_written, 'path': str(out_path), 'mode': 'single'}

    fn = {'shards': _export_shards, 'split': _export_split}.get(req.mode, _export)
    job = entry.worker.submit('export', fn, req.model_dump())
    return {'job_id': job.id}


# ---- stats / metadata index ----------------------------------------------

@app.post("/api/datasets/{dataset_id}/stats")
def stats(dataset_id: str, req: StatsRequest):
    entry = _entry(dataset_id)
    explorer = _explorer(dataset_id)

    def _stats():
        return explorer.get_statistics(req.field or None)

    job = entry.worker.submit('stats', _stats, req.model_dump())
    return {'job_id': job.id}


class MetaFieldsRequest(BaseModel):
    rebuild: bool = False    # force a from-scratch index build (full corpus pass)


@app.post("/api/datasets/{dataset_id}/meta/fields")
def meta_fields(dataset_id: str, req: Optional[MetaFieldsRequest] = None):
    entry = _entry(dataset_id)
    explorer = _explorer(dataset_id)
    if not dx.METAQUERY_AVAILABLE:
        raise HTTPException(501, "metaquery.py / metaindex.py not available")
    rebuild = bool(req and req.rebuild)

    def _fields():
        idx = explorer.get_metaindex(rebuild=rebuild)
        return {'n_rows': idx.n_rows, 'fields': idx.field_summary()}

    job = entry.worker.submit('meta-rebuild' if rebuild else 'meta-fields',
                              _fields, {'rebuild': rebuild})
    return {'job_id': job.id}


@app.post("/api/datasets/{dataset_id}/meta/values")
def meta_values(dataset_id: str, req: MetaValuesRequest):
    entry = _entry(dataset_id)
    explorer = _explorer(dataset_id)
    if not dx.METAQUERY_AVAILABLE:
        raise HTTPException(501, "metaquery.py / metaindex.py not available")

    def _values():
        idx = explorer.get_metaindex()
        vc = idx.value_counts(req.field, top=req.top)
        return {'field': req.field,
                'values': [{'value': str(v), 'count': int(c)} for v, c in vc.items()]}

    job = entry.worker.submit('meta-values', _values, req.model_dump())
    return {'job_id': job.id}


# ---- near-duplicates ------------------------------------------------------

@app.post("/api/datasets/{dataset_id}/neardupe")
def neardupe(dataset_id: str, req: NeardupeRequest):
    entry = _entry(dataset_id)
    explorer = _explorer(dataset_id)
    if not dx.NEARDUPE_AVAILABLE:
        raise HTTPException(501, "neardupe.py not available")
    if req.perms & (req.perms - 1):
        raise HTTPException(400, f"perms must be a power of two (got {req.perms})")

    def _neardupe():
        clusters = explorer.find_near_duplicates(
            threshold=req.threshold, ngram=req.ngram, perms=req.perms,
            field=req.field, sample=req.sample, device=req.device,
            exact_cardinality=req.exact_cardinality, rebuild=req.rebuild,
            min_tokens=req.min_tokens, set_name=req.set_name)
        return {'n_clusters': len(clusters or [])}

    job = entry.worker.submit('neardupe', _neardupe, req.model_dump())
    return {'job_id': job.id}


@app.get("/api/datasets/{dataset_id}/neardupe/clusters")
def neardupe_clusters(dataset_id: str, top: int = 100):
    explorer = _explorer(dataset_id)
    state = explorer.dupe_state
    if not state or not state.get('clusters'):
        return {'clusters': []}
    out = []
    for i, c in enumerate(state['clusters'][:top]):
        out.append(_jsonable({
            'rank': i + 1, 'size': c['size'],
            'max_jaccard': c.get('max_sim'), 'mean_jaccard': c.get('mean_sim'),
            'max_containment': c.get('max_containment'),
            'members': list(c['members'])[:50],
        }))
    return {'clusters': out, 'n_total': len(state['clusters'])}


@app.post("/api/datasets/{dataset_id}/neardupe/prune")
def neardupe_prune(dataset_id: str, req: PruneRequest):
    entry = _entry(dataset_id)
    explorer = _explorer(dataset_id)
    if not explorer.dupe_state or not explorer.dupe_state.get('clusters'):
        raise HTTPException(409, "No near-duplicate results yet; run neardupe first")

    def _prune():
        result = explorer.prune_near_duplicates(
            req.out_dir, write=req.write, keep=req.keep,
            include_chains=req.include_chains, include_short=req.include_short,
            protect_val=req.protect_val)
        return {'removed': len(result['kill']), 'written': result['written'],
                'contaminated': result['contaminated'], 'shards': result['shards']}

    job = entry.worker.submit('prune', _prune, req.model_dump())
    return {'job_id': job.id}


# ---- registry (managed datasets) ------------------------------------------

@app.get("/api/config")
def get_config():
    return {'tokenized_root': TOKENIZED_ROOT}


@app.get("/api/registry")
def registry_list():
    entries = _registry().list()
    open_paths = {str(Path(e.path).resolve()): e.id
                  for e in DATASETS.values() if e.status == 'ready'}
    for e in entries:
        try:
            e['open_as'] = open_paths.get(str(Path(e['path']).resolve()))
            e['path_exists'] = Path(e['path']).exists()
        except OSError:
            e['open_as'], e['path_exists'] = None, False
    return {'registry_path': str(_registry().path), 'datasets': entries}


@app.post("/api/registry")
def registry_add(req: RegisterRequest):
    reg = _registry()
    path = Path(req.path).expanduser()
    if not path.exists():
        raise HTTPException(400, f"Path does not exist: {path}")
    existing = reg.find_by_path(path)
    if existing:
        raise HTTPException(409, f"Already registered as {existing['id']!r} "
                                 f"({existing['name']})")
    stats = _quick_stats(path)
    entry = {
        'id': reg.unique_id(req.name),
        'name': req.name,
        # absolute(), not resolve(): caches are keyed on the path spelling, and
        # resolve() rewrites mapped drives/symlinks to a different spelling. Store
        # what the user gave us so Library opens hit the same caches they adopt.
        'path': str(path.absolute()),
        'kind': req.kind or stats.pop('kind_guess'),
        'tags': req.tags,
        'notes': req.notes,
        'derived_from': req.derived_from,
        'open_opts': {
            'text_field': req.text_field, 'tok_kind': req.tok_kind,
            'tok_path': req.tok_path, 'special_tokens': req.special_tokens,
            'raw_shards': req.raw_shards, 'recursive': req.recursive,
        },
        'stats': {k: v for k, v in stats.items() if k != 'kind_guess'},
        'created': time.time(),
        'last_opened': None,
    }
    reg.upsert(entry)
    return entry


class RegisterLoadedRequest(BaseModel):
    name: str
    tags: List[str] = []
    notes: str = ''


@app.post("/api/datasets/{dataset_id}/register")
def register_loaded(dataset_id: str, req: RegisterLoadedRequest):
    """Add an already-loaded dataset to the library, using the live instance as
    the source of truth: exact path, the open options it was loaded with, and
    metadata-accurate stats."""
    ds = _entry(dataset_id)
    explorer = _explorer(dataset_id)
    reg = _registry()
    path = Path(ds.path)
    existing = reg.find_by_path(path)
    if existing:
        raise HTTPException(409, f"Already registered as {existing['id']!r} "
                                 f"({existing['name']})")
    opts = ds.opts or {}
    md = explorer.metadata
    # The live explorer already KNOWS the vitals -- never re-stat the tree
    # synchronously (a 5,843-file SMB sweep once froze this endpoint for a
    # minute). Only the manifest token count needs a (cheap) disk look.
    stats = {'num_rows': md.get('num_rows'),
             'size_mb': md.get('file_size'),
             'num_files': md.get('num_files', 1),
             'tokens': _manifest_stats(path)['tokens']}
    entry = {
        'id': reg.unique_id(req.name),
        'name': req.name,
        'path': str(path.absolute()),
        'kind': 'tokenized' if (md.get('is_token_shard')
                                or explorer.file_type == 'npy') else 'text',
        'tags': req.tags,
        'notes': req.notes,
        'derived_from': None,
        'open_opts': {k: opts.get(k) for k in
                      ('text_field', 'tok_kind', 'tok_path',
                       'special_tokens', 'raw_shards', 'recursive')},
        'stats': _jsonable(stats),
        'created': time.time(),
        'last_opened': time.time(),
    }
    reg.upsert(entry)
    return entry


@app.patch("/api/registry/{rid}")
def registry_edit(rid: str, req: RegistryEditRequest):
    reg = _registry()
    entry = reg.get(rid)
    changes = {k: v for k, v in req.model_dump().items() if v is not None}
    open_keys = ('text_field', 'tok_kind', 'tok_path', 'special_tokens',
                 'raw_shards', 'recursive')
    for k in list(changes):
        if k in open_keys:
            entry.setdefault('open_opts', {})[k] = changes.pop(k)
    entry.update(changes)
    reg.upsert(entry)
    return entry


@app.delete("/api/registry/{rid}")
def registry_remove(rid: str):
    _registry().delete(rid)     # metadata only; data files are never touched
    return {'unregistered': rid}


@app.post("/api/registry/{rid}/open")
def registry_open(rid: str):
    entry = _registry().get(rid)
    # Already open? Just point the caller at the live instance.
    for ds in DATASETS.values():
        try:
            if (ds.status == 'ready'
                    and str(Path(ds.path).resolve()) == str(Path(entry['path']).resolve())):
                return {'dataset_id': ds.id, 'job_id': None, 'already_open': True}
        except OSError:
            pass
    oo = entry.get('open_opts') or {}
    ds_entry = _start_load(LoadRequest(
        path=entry['path'], text_field=oo.get('text_field'),
        tok_kind=oo.get('tok_kind'), tok_path=oo.get('tok_path'),
        special_tokens=oo.get('special_tokens'),
        raw_shards=bool(oo.get('raw_shards')),
        recursive=bool(oo.get('recursive'))))
    return {'dataset_id': ds_entry.id, 'job_id': ds_entry.load_job.id}


@app.get("/api/registry/scan")
def registry_scan(path: str):
    """Registerable candidates in ONE directory (non-recursive, by design)."""
    p = Path(path).expanduser()
    if not p.is_dir():
        raise HTTPException(400, f"Not a directory: {p}")
    reg = _registry()
    registered = {str(Path(e['path']).resolve()) for e in reg.list()}
    out = []
    # scandir for type info (see /api/browse): per-entry Path stats over SMB
    # turn big directories into minutes of silence.
    with os.scandir(p) as it:
        scan_children = sorted(it, key=lambda e: e.name.lower())
    for de in scan_children:
        if de.name.startswith('.') or de.name == 'tmp':
            continue
        child = p / de.name
        cand = None
        try:
            child_is_dir = de.is_dir()
        except OSError:
            continue
        if child_is_dir:
            try:
                inner = [q for q in itertools.islice(child.iterdir(), 500)
                         if _is_data_file(q)]
            except PermissionError:
                continue
            if inner:
                npys = [q for q in inner if q.suffix.lower() == '.npy']
                has_manifest = any(child.glob('manifest_*.json'))
                cand = {'path': str(child), 'name': child.name,
                        'kind': 'tokenized' if (has_manifest or npys) else 'text',
                        'files': len(npys) if npys else len(inner)}
        elif _is_data_file(child):
            cand = {'path': str(child), 'name': child.stem,
                    'kind': 'tokenized' if child.suffix.lower() == '.npy' else 'text',
                    'files': 1}
        if cand:
            cand['registered'] = str(Path(cand['path']).resolve()) in registered
            out.append(cand)
    return {'path': str(p), 'candidates': out[:300]}


# ---- filters: rule engine --------------------------------------------------

FILTER_RULE_KINDS = ('contains', 'startswith', 'len_lt', 'len_gt', 'regex', 'python')

# The python escape hatch runs with these names only (LAN tool, single user;
# an honest eval beats a pretend-safe DSL, but builtins stay off the table).
_RULE_ENV = {'__builtins__': {}, 're': re, 'len': len, 'str': str, 'int': int,
             'float': float, 'any': any, 'all': all, 'min': min, 'max': max,
             'sum': sum, 'abs': abs}


class FilterRule(BaseModel):
    name: str
    kind: str                       # one of FILTER_RULE_KINDS
    field: Optional[str] = None     # None = the dataset's resolved text field
    needle: Optional[str] = None    # contains
    first_n: Optional[int] = None   # contains: only look in the first N chars
    prefix: Optional[str] = None    # startswith
    value: Optional[int] = None     # len_lt / len_gt (chars)
    pattern: Optional[str] = None   # regex (search)
    expr: Optional[str] = None      # python: eval'd with rec / re / len ...


class ScrubDef(BaseModel):
    """An ordered regex rewrite applied to SURVIVING records (post-drop).
    Scrubs run in list order on the named field (default: the dataset's text
    field). Applied at composition time — file mode rewrites the intermediate,
    stream mode ships them in the view manifest and pre_tokenize applies them
    in-stream. Never mutates source data."""
    name: str
    pattern: str
    replacement: str = ''
    field: Optional[str] = None     # top-level field only (body text)
    literal: bool = False           # treat pattern as an exact substring
                                    # (re.escape'd at compile — paste-safe,
                                    # no metacharacter knowledge needed)
    glob: bool = False              # exact substring EXCEPT '*' = shortest
                                    # stretch of anything within the line.
                                    # Separate from literal on purpose: '*'
                                    # is a real character in wiki text
                                    # ("(* 1305; † 1345)" birth-stars).
    line: bool = False              # the match selects its ENTIRE line: the
                                    # replacement applies to the whole line
                                    # incl. its newline (blank = delete line)
    nocase: bool = False            # case-insensitive matching (any mode;
                                    # for regex it's sugar for a (?i) prefix)
    escapes: bool = False           # literal/glob: interpret \n \t \\ in the
                                    # pattern AND replacement (opt-in — the
                                    # paste-anything contract stays default)


_SCRUB_ESCAPES = {'n': '\n', 't': '\t', '\\': '\\'}


def _decode_scrub_escapes(s: str, name: str) -> str:
    out = []
    i = 0
    while i < len(s):
        if s[i] == '\\':
            if i + 1 >= len(s):
                raise HTTPException(400, f"Scrub {name!r}: trailing backslash "
                                         "(with escapes on, write \\\\ for a literal one)")
            nxt = s[i + 1]
            if nxt not in _SCRUB_ESCAPES:
                raise HTTPException(400, f"Scrub {name!r}: unknown escape "
                                         f"\\{nxt} (known: \\n \\t \\\\)")
            out.append(_SCRUB_ESCAPES[nxt])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return ''.join(out)


def _scrub_regex_source(sd: 'ScrubDef') -> str:
    """The final regex source a scrub compiles to. Literal escapes wholesale;
    glob escapes the pieces between '*'s and joins them with a lazy
    within-line wildcard ([^\\n]*? -- shortest match, so deletions take the
    minimum span); line mode wraps in line-greedy guards ([^\\n] can't cross
    newlines, so no MULTILINE flag is involved). Single authority -- the
    compiler AND the view-manifest resolver both use it, so stream and file
    modes can't drift."""
    if sd.literal and sd.glob:
        raise HTTPException(400, f"Scrub {sd.name!r}: literal and glob are "
                                 "mutually exclusive modes")
    pattern = sd.pattern
    if sd.escapes and (sd.literal or sd.glob):
        pattern = _decode_scrub_escapes(pattern, sd.name)
        if sd.line and '\n' in pattern:
            raise HTTPException(400, f"Scrub {sd.name!r}: line mode matches "
                                     "within a single line — a \\n in the "
                                     "pattern can never match there")
    if sd.glob:
        src = r"[^\n]*?".join(re.escape(part) for part in pattern.split('*'))
    elif sd.literal:
        src = re.escape(pattern)
    else:
        src = sd.pattern
    flags = ''
    if sd.line:
        # Anchored form: attempt only at line starts (^ with m-flag fails
        # O(1) elsewhere), test the line ONCE via lookahead, then consume it
        # unconditionally. The naive [^\n]*(?:…)[^\n]*\n? wrapper re-attempts
        # at every offset with per-split backtracking — O(L²) per
        # non-matching line, ~180ms on a single 20KB line (513x slower),
        # and this corpus has 587KB records. Semantics identical (verified).
        src = r"^(?=[^\n]*?(?:" + src + r"))[^\n]*\n?"
        flags += 'm'
    if sd.nocase:
        flags += 'i'
    if flags:
        # inline flags must lead the pattern — composed here, once
        src = f"(?{flags})" + src
    return src


def _scrub_repl_source(sd: 'ScrubDef') -> str:
    """The final replacement string. re.subn processes \\1/\\g escapes in the
    replacement -- wanted in regex mode (backrefs), a paste hazard in
    literal/glob mode (a verbatim 'C:\\Users' would be a bad-escape error).
    Paste-safe modes are paste-safe END TO END: backslashes are literalized."""
    if sd.literal or sd.glob:
        repl = sd.replacement
        if sd.escapes:
            repl = _decode_scrub_escapes(repl, sd.name)
        return repl.replace('\\', '\\\\')
    return sd.replacement


class FilterDef(BaseModel):
    name: str
    rules: List[FilterRule]
    notes: str = ''
    scrubs: List[ScrubDef] = []
    # Re-run the scrub chain until nothing changes (capped): for chains whose
    # rules feed each other (a rewrite exposing a pattern an EARLIER scrub
    # would have caught). Part of the composition's identity.
    scrub_fixpoint: bool = False
    id: Optional[str] = None            # set on update; minted on create


SCRUB_MAX_PASSES = 10       # fixpoint cap: oscillating/growing chains stop here


def _rules_hash(fdef_or_rules) -> str:
    """8-hex fingerprint of a filter's RULES ONLY (scrubs excluded).
    Materialized sets are stamped with it, so scrub-only edits (which bump the
    filter version) do not invalidate drop sets that are still exact."""
    rules = fdef_or_rules.get('rules') if isinstance(fdef_or_rules, dict) \
        else fdef_or_rules
    blob = json.dumps(rules, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(blob.encode('utf-8')).hexdigest()[:8]


_PATTERN_STRESS_OK: set = set()    # per-process cache of vetted patterns

_PATTERN_STRESS_CODE = r"""
import re, sys
pat = re.compile(sys.argv[1])
line = 'word ' * 12
docs = [
    'x' * 20000,
    '\n'.join([line] * 600),
    'prose ' * 4000 + '\nReferences\n' + '\n'.join([line] * 200),   # long tail
    'a(b c(d ' * 3000,                                              # paren soup
]
for d in docs:
    pat.subn('', d)
print('ok')
"""


def _stress_test_pattern(name: str, pattern: str, timeout_s: float = 2.0):
    """Reject regexes that hang on plausible documents (catastrophic
    backtracking). A pathological pattern inside subn() blocks the dataset
    worker in C code where even cancellation can't reach -- so it must be
    caught HERE, in a killable subprocess, before any job runs it."""
    if pattern in _PATTERN_STRESS_OK:
        return
    try:
        r = subprocess.run([sys.executable, '-c', _PATTERN_STRESS_CODE, pattern],
                           capture_output=True, timeout=timeout_s)
        if r.returncode == 0:
            _PATTERN_STRESS_OK.add(pattern)
    except subprocess.TimeoutExpired:
        raise HTTPException(
            400, f"Scrub {name!r}: pattern exhibits catastrophic backtracking "
                 f"(hung >{timeout_s:.0f}s on a stress document). Rework it -- "
                 f"common fix: make repeated line-groups consume a MANDATORY "
                 f"newline, e.g. '(?:[^\\n]*\\n){{0,40}}' not '(?:[^\\n]*\\n?){{0,40}}'.")


def _compile_scrubs(scrubs, default_field: Optional[str]):
    """[(name, field, compiled_pattern, replacement)], validated."""
    out = []
    for s in scrubs:
        s = ScrubDef(**s) if isinstance(s, dict) else s
        field = s.field or default_field
        if field and '.' in field:
            raise HTTPException(400, f"Scrub {s.name!r}: dotted fields are not "
                                     "supported for scrubs (body text is top-level)")
        src = _scrub_regex_source(s)
        try:
            pat = re.compile(src)
        except re.error as e:
            raise HTTPException(400, f"Scrub {s.name!r}: bad regex: {e}")
        if not s.literal:
            # escaped literals have no quantifiers — provably linear, skip vet
            # (vet the FINAL source so line-wrapping is what gets stress-tested)
            _stress_test_pattern(s.name, src)
        out.append((s.name, field, pat, _scrub_repl_source(s)))
    return out


def _apply_scrubs(rec: Dict[str, Any], compiled,
                  counts: Optional[Dict[str, List[int]]] = None,
                  fixpoint: bool = False):
    """Apply compiled scrubs to rec in place; with fixpoint, repeat the whole
    chain until a full pass changes nothing (capped at SCRUB_MAX_PASSES).
    Returns (changed, passes, converged, hit_names) -- hit_names is the set
    of scrub names that touched this record (union across passes), feeding
    the preview's per-record attribution masks. counts[name] accumulates
    [docs_hit, total_subs, chars_delta] -- docs counted once per record even
    across passes."""
    changed_any = False
    hit_names = set()
    passes = 0
    while True:
        passes += 1
        pass_changed = False
        for name, field, pat, repl in compiled:
            v = rec.get(field)
            if not isinstance(v, str):
                continue
            new, n = pat.subn(repl, v)
            if n:
                rec[field] = new
                pass_changed = True
                if counts is not None:
                    c = counts.setdefault(name, [0, 0, 0])
                    if name not in hit_names:
                        c[0] += 1
                    c[1] += n
                    c[2] += len(v) - len(new)
                hit_names.add(name)
        changed_any = changed_any or pass_changed
        if not pass_changed:
            return changed_any, passes, True, hit_names
        if not fixpoint:
            return changed_any, passes, True, hit_names
        if passes >= SCRUB_MAX_PASSES:
            return changed_any, passes, False, hit_names


def _compile_rule(rule: FilterRule, default_field: Optional[str]):
    """Rule -> callable(rec) -> bool. Compiled once per evaluation, applied
    per record. A missing field never matches (safe default for drops).
    Dotted field names traverse nested dicts (`meta.title`), same convention
    as the tokenize extraction templates; an exact flat key wins first."""
    field = rule.field or default_field

    def val(rec) -> str:
        if not field:
            return ''
        v = rec.get(field)
        if v is None and '.' in field:
            v = rec
            for part in field.split('.'):
                if not isinstance(v, dict):
                    v = None
                    break
                v = v.get(part)
        return '' if v is None or isinstance(v, (dict, list)) else str(v)

    k = rule.kind
    if k == 'contains':
        needle, first_n = rule.needle or '', rule.first_n
        if first_n:
            return lambda rec: needle in val(rec)[:first_n]
        return lambda rec: needle in val(rec)
    if k == 'startswith':
        prefix = rule.prefix or ''
        return lambda rec: val(rec).startswith(prefix)
    if k == 'len_lt':
        n = int(rule.value or 0)
        return lambda rec: len(val(rec)) < n
    if k == 'len_gt':
        n = int(rule.value or 0)
        return lambda rec: len(val(rec)) > n
    if k == 'regex':
        pat = re.compile(rule.pattern or '')
        return lambda rec: pat.search(val(rec)) is not None
    if k == 'python':
        code = compile(rule.expr or 'False', f'<rule {rule.name}>', 'eval')
        return lambda rec: bool(eval(code, _RULE_ENV, {'rec': rec}))
    raise HTTPException(400, f"Unknown rule kind {k!r} "
                             f"(valid: {', '.join(FILTER_RULE_KINDS)})")


def _validate_filter(f: FilterDef):
    if not f.rules:
        raise HTTPException(400, "A filter needs at least one rule")
    seen = set()
    for r in f.rules:
        if not r.name or r.name in seen:
            raise HTTPException(400, f"Rule names must be unique and non-empty "
                                     f"(problem: {r.name!r})")
        seen.add(r.name)
        _compile_rule(r, 'text')     # raises on bad kind/regex/expr
    sseen = set()
    for s in f.scrubs:
        if not s.name or s.name in sseen:
            raise HTTPException(400, f"Scrub names must be unique and non-empty "
                                     f"(problem: {s.name!r})")
        sseen.add(s.name)
    _compile_scrubs(f.scrubs, 'text')   # raises on bad regex / dotted field


@app.get("/api/filters")
def filters_list():
    filters = _registry().list_filters()
    for f in filters:
        f['rules_hash'] = _rules_hash(f)    # for set-freshness checks (UI + compose)
    return {'filters': filters}


@app.post("/api/filters")
def filters_upsert(f: FilterDef):
    _validate_filter(f)
    reg = _registry()
    entry = f.model_dump()
    if f.id:
        existing = reg.get_filter(f.id)
        entry['id'] = f.id
        entry['version'] = int(existing.get('version', 1)) + 1
        entry['created'] = existing.get('created')
    else:
        entry['id'] = f"flt-{re.sub(r'[^a-z0-9]+', '-', f.name.lower()).strip('-')}"
        entry['version'] = 1
        entry['created'] = time.time()
    entry['updated'] = time.time()
    reg.upsert_filter(entry)
    return entry


@app.get("/api/filters/{fid}/evals")
def filter_evals_list(fid: str):
    _registry().get_filter(fid)          # 404 on unknown filter
    return {'evals': _registry().list_filter_evals(fid)}


@app.delete("/api/filters/{fid}/evals")
def filter_evals_clear(fid: str):
    _registry().clear_filter_evals(fid)
    return {'cleared': fid}


@app.delete("/api/filters/{fid}")
def filters_delete(fid: str):
    _registry().delete_filter(fid)
    return {'deleted': fid}


class FilterEvalRequest(BaseModel):
    sample: Optional[int] = 10_000   # None = full corpus scan
    materialize: bool = False        # store per-rule sets (full scan only)


@app.post("/api/datasets/{dataset_id}/filters/{fid}/evaluate")
def filter_evaluate(dataset_id: str, fid: str, req: FilterEvalRequest):
    """Evaluate a filter against a corpus.

    Shallow (sampled): per-rule counts and per-10k rates -- the corpus A/B
    table. Materialize (full scan): additionally stores each rule's matches as
    an ordinary set, so drops are BROWSABLE before they're destructive, and
    the combined set feeds export/tokenize exclusion.
    """
    entry = _entry(dataset_id)
    explorer = _explorer(dataset_id)
    fdef = _registry().get_filter(fid)
    [FilterRule(**r) for r in fdef['rules']]          # validate before queueing
    if req.materialize and req.sample:
        raise HTTPException(400, "materialize requires a full scan (sample=null)")

    job = entry.worker.submit(
        'filter-eval',
        lambda: _filter_eval_impl(explorer, fdef, req.sample, req.materialize),
        {'filter': fid, 'sample': req.sample, 'materialize': req.materialize})
    return {'job_id': job.id}


def _scrub_diff_regions(before_text: str, scrubs, field: str, ctx: int = 70,
                        max_regions: int = 8, fixpoint: bool = False):
    """Every change a chain makes to one document, as per-match regions with
    context, attributed to the scrub that made them. Walks the chain exactly
    as execution does (each scrub sees its predecessors' output), so what you
    read is what will happen. Replaces the old flat first-difference excerpt,
    which anchored on the FIRST changed byte and routinely hid the
    substantive change (a tail chop at doc end) behind a trivial one (a
    2-char husk deletion at doc top)."""
    regions = []
    text = before_text
    passes = SCRUB_MAX_PASSES if fixpoint else 1
    for _ in range(passes):
        changed_pass = False
        for name, f, pat, repl in scrubs:
            if f != field:
                continue
            hit = False
            for m in pat.finditer(text):
                hit = True
                if len(regions) < max_regions:
                    regions.append({
                        'scrub': name,
                        'ctx_before': text[max(0, m.start() - ctx):m.start()],
                        'removed': m.group(0)[:400],
                        'added': m.expand(repl)[:400],
                        'ctx_after': text[m.end():m.end() + ctx],
                    })
            if hit:
                text = pat.sub(repl, text)
                changed_pass = True
        if not changed_pass or not fixpoint:
            break
    return regions


class TransformDef(BaseModel):
    """The fourth noun: a corpus-agnostic ordered rewrite chain. A filter is
    a predicate (composes as set algebra); a transform is a function
    (composes by sequencing, optionally to fixpoint). Never materializes on
    its own -- only as part of a composition at an export pass."""
    name: str
    scrubs: List[ScrubDef]
    fixpoint: bool = False
    notes: str = ''
    id: Optional[str] = None


def _validate_transform(t: TransformDef):
    if not t.name.strip():
        raise HTTPException(400, "A transform needs a name")
    if not t.scrubs:
        raise HTTPException(400, "A transform needs at least one scrub")
    seen = set()
    for s in t.scrubs:
        if not s.name or s.name in seen:
            raise HTTPException(400, f"Scrub names must be unique and non-empty "
                                     f"(problem: {s.name!r})")
        seen.add(s.name)
    _compile_scrubs(t.scrubs, 'text')


@app.get("/api/transforms")
def transforms_list():
    return {'transforms': _registry().list_transforms()}


@app.post("/api/transforms")
def transforms_upsert(t: TransformDef):
    _validate_transform(t)
    reg = _registry()
    entry = t.model_dump()
    if t.id:
        existing = reg.get_transform(t.id)
        entry['id'] = t.id
        entry['version'] = int(existing.get('version', 1)) + 1
        entry['created'] = existing.get('created')
    else:
        entry['id'] = f"tfm-{re.sub(r'[^a-z0-9]+', '-', t.name.lower()).strip('-')}"
        entry['version'] = 1
        entry['created'] = time.time()
    entry['updated'] = time.time()
    reg.upsert_transform(entry)
    return entry


@app.delete("/api/transforms/{tid}")
def transforms_delete(tid: str):
    _registry().delete_transform(tid)
    return {'deleted': tid}


@app.get("/api/transforms/{tid}/evals")
def transform_evals_list(tid: str):
    _registry().get_transform(tid)
    return {'evals': _registry().list_transform_evals(tid)}


class TransformPreviewRequest(BaseModel):
    sample: int = 10_000
    examples: int = 8
    filter_id: Optional[str] = None    # preview on that filter's SURVIVORS


@app.post("/api/datasets/{dataset_id}/transforms/{tid}/preview")
def transform_preview(dataset_id: str, tid: str, req: TransformPreviewRequest):
    """Dry-run a transform chain on a sample (optionally the survivors of a
    filter): per-scrub hit rates + chars removed + before/after diffs.
    Nothing is written -- this is the eyeball gate before composing."""
    entry = _entry(dataset_id)
    explorer = _explorer(dataset_id)
    tdef = _registry().get_transform(tid)
    fdef = _registry().get_filter(req.filter_id) if req.filter_id else None

    def _preview():
        default_field = explorer._resolve_text_field(None)
        rules = [_compile_rule(FilterRule(**r), default_field)
                 for r in (fdef['rules'] if fdef else [])]
        scrubs = _compile_scrubs(tdef['scrubs'], default_field)
        fixpoint = bool(tdef.get('fixpoint'))
        total = int(explorer.metadata.get('num_rows') or 0)
        n = min(req.sample, total) if total else req.sample
        picks = np.unique(np.linspace(0, max(total - 1, 0), n).astype(np.int64))
        print(f"Transform preview: {tdef['name']!r} v{tdef.get('version', 1)} on "
              f"{picks.size:,} sampled record(s)"
              + (f", survivors of filter {fdef['name']!r}" if fdef else "")
              + (f" (fixpoint mode, cap {SCRUB_MAX_PASSES})" if fixpoint else ""))
        counts: Dict[str, List[int]] = {}
        examples: List[Dict[str, Any]] = []
        changed_idx: List[int] = []
        changed_masks: List[int] = []
        scrub_bit = {name: 1 << i for i, (name, _f, _p, _r) in enumerate(scrubs)}
        n_survivors = 0
        n_changed = 0
        max_passes = 0
        n_nonconverged = 0
        for i, gi in enumerate(picks):
            rec = _get_record_dict(explorer, int(gi))
            dropped = False
            for fn in rules:
                try:
                    if fn(rec):
                        dropped = True
                        break
                except Exception:
                    pass
            if not dropped:
                n_survivors += 1
                before = {f: rec.get(f) for _n, f, _p, _r in scrubs}
                changed, passes, converged, hits = _apply_scrubs(
                    rec, scrubs, counts, fixpoint=fixpoint)
                max_passes = max(max_passes, passes)
                if not converged:
                    n_nonconverged += 1
                if changed:
                    n_changed += 1
                    changed_idx.append(int(gi))
                    changed_masks.append(sum(scrub_bit[n2] for n2 in hits))
                    if len(examples) < req.examples:
                        for _n, f, _p, _r in scrubs:
                            b, a = before.get(f), rec.get(f)
                            if isinstance(b, str) and b != a:
                                examples.append({
                                    'index': int(gi), 'field': f,
                                    'regions': _scrub_diff_regions(
                                        b, scrubs, f, fixpoint=fixpoint)})
                                break
            if (i + 1) % 500 == 0:
                dx.report_progress('scrub preview', i + 1, int(picks.size),
                                   note=f"{n_changed:,} changed")
        per10k = lambda c: round(c / max(n_survivors, 1) * 10_000, 1)
        out_scrubs = []
        for name, _f, _p, _r in scrubs:
            c = counts.get(name, [0, 0, 0])
            out_scrubs.append({'name': name, 'docs': c[0],
                               'docs_per_10k': per10k(c[0]),
                               'subs': c[1], 'chars_removed': c[2]})
            print(f"  {name:<20} {c[0]:>7,} docs  {per10k(c[0]):>8.1f}/10k  "
                  f"{c[1]:>8,} subs  {c[2]:>10,} chars removed")
        print(f"  {'ANY CHANGE':<20} {n_changed:>7,} docs  {per10k(n_changed):>8.1f}/10k")
        if fixpoint:
            print(f"  fixpoint: max {max_passes} pass(es)"
                  + (f", {n_nonconverged:,} record(s) DID NOT CONVERGE at the "
                     f"cap -- the chain oscillates or grows; inspect it"
                     if n_nonconverged else " -- all records converged"))
        # every changed record's index + per-record scrub bitmask (bit i =
        # scrub i in chain order hit it), evenly thinned past 5,000 in
        # lockstep — the UI pages/filters these for damage spot-checking
        if len(changed_idx) > 5000:
            step = -(-len(changed_idx) // 5000)
            changed_idx = changed_idx[::step]
            changed_masks = changed_masks[::step]
        result = {'transform': tdef['name'], 'version': tdef.get('version', 1),
                  'filter': fdef['name'] if fdef else None,
                  'sampled': int(picks.size), 'survivors': n_survivors,
                  'changed': n_changed, 'changed_per_10k': per10k(n_changed),
                  'fixpoint': fixpoint, 'max_passes': max_passes,
                  'nonconverged': n_nonconverged,
                  'scrubs': out_scrubs, 'examples': examples,
                  'changed_indices': changed_idx,
                  'changed_masks': changed_masks}
        # record for the per-corpus preview history (same pattern as filter evals)
        if REGISTRY is not None and tdef.get('id'):
            try:
                parent = REGISTRY.find_by_path(explorer.original_filepath)
                REGISTRY.record_transform_eval(tdef['id'], {
                    'dataset': (parent['name'] if parent
                                else explorer.original_filepath.name),
                    'path': str(explorer.original_filepath.absolute()),
                    'version': result['version'],
                    'filter': result['filter'],
                    'sampled': result['sampled'], 'survivors': n_survivors,
                    'changed': n_changed, 'changed_per_10k': result['changed_per_10k'],
                    'nonconverged': n_nonconverged,
                    'scrubs': out_scrubs,
                    'created': time.time(),
                })
            except Exception as e:
                print(f"(preview history not recorded: {e})")
        return result

    job = entry.worker.submit('transform-preview', _preview,
                              {'transform': tid, 'filter': req.filter_id,
                               'sample': req.sample})
    return {'job_id': job.id}


def _full_diff_segments(before: str, after: str, cap: int = 400_000):
    """The ENTIRE document as eq/del/ins segments — the expand view. A real
    sequence diff of the end states (so multi-scrub/multi-pass layering never
    needs representing); per-scrub attribution stays in the region list.
    Docs beyond `cap` skip highlighting (SequenceMatcher worst case is
    quadratic) and render as plain after-text."""
    if len(before) + len(after) > cap:
        return [{'t': 'eq', 's': after}], False
    import difflib
    sm = difflib.SequenceMatcher(None, before, after, autojunk=False)
    segs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            segs.append({'t': 'eq', 's': before[i1:i2]})
        else:
            if i2 > i1:
                segs.append({'t': 'del', 's': before[i1:i2]})
            if j2 > j1:
                segs.append({'t': 'ins', 's': after[j1:j2]})
    return segs, True


class TransformDiffRequest(BaseModel):
    indices: List[int]              # records to render (from changed_indices)
    full: bool = False              # also return full-document segments


@app.post("/api/datasets/{dataset_id}/transforms/{tid}/diff")
def transform_diff(dataset_id: str, tid: str, req: TransformDiffRequest):
    """Diff regions for specific records, on demand — the paging companion to
    the preview. Synchronous: a handful of fast-path record reads plus regex
    on ≤32 docs; no job, no corpus walk, so 'next' clicks stay snappy."""
    explorer = _explorer(dataset_id)
    tdef = _registry().get_transform(tid)
    if len(req.indices) > 32:
        raise HTTPException(400, "At most 32 indices per diff page")
    default_field = explorer._resolve_text_field(None)
    scrubs = _compile_scrubs(tdef['scrubs'], default_field)
    fixpoint = bool(tdef.get('fixpoint'))
    out = []
    fields = list(dict.fromkeys(f for _n, f, _p, _r in scrubs))
    for gi in req.indices:
        rec = _get_record_dict(explorer, int(gi))
        for f in fields:
            v = rec.get(f)
            if isinstance(v, str):
                regions = _scrub_diff_regions(v, scrubs, f, fixpoint=fixpoint)
                if regions:
                    ex = {'index': int(gi), 'field': f, 'regions': regions}
                    if req.full:
                        work = dict(rec)
                        _apply_scrubs(work, scrubs, fixpoint=fixpoint)
                        segs, highlighted = _full_diff_segments(v, work.get(f, ''))
                        ex['segments'] = segs
                        ex['segments_highlighted'] = highlighted
                    out.append(ex)
                    break
    return {'examples': out}


def _filter_eval_impl(explorer, fdef: Dict[str, Any],
                      sample: Optional[int], materialize: bool):
    """The filter-evaluation job body. Module-level so the composed-tokenize
    job can run the materialize phase inline on its own worker thread."""
    rules = [FilterRule(**r) for r in fdef['rules']]
    default_field = explorer._resolve_text_field(None)
    compiled = [(r.name, _compile_rule(r, default_field)) for r in rules]
    total = int(explorer.metadata.get('num_rows') or 0)
    counts = {name: 0 for name, _ in compiled}
    matches = {name: [] for name, _ in compiled} if materialize else None
    any_count = 0
    lengths: List[int] = []
    n_eval = 0

    if sample and total > sample:
        picks = np.unique(np.linspace(0, total - 1, sample).astype(np.int64))
        print(f"Evaluating filter {fdef['name']!r} v{fdef.get('version', 1)} "
              f"on {picks.size:,} of {total:,} records (evenly spaced sample)")
        record_iter = ((int(gi), _get_record_dict(explorer, int(gi)))
                       for gi in picks)
        n_total = int(picks.size)
    else:
        print(f"Evaluating filter {fdef['name']!r} v{fdef.get('version', 1)} "
              f"on ALL {total:,} records")
        record_iter = explorer._iter_records_streaming()
        n_total = total

    for gi, rec in record_iter:
        hit_any = False
        for name, fn in compiled:
            try:
                hit = fn(rec)
            except Exception:
                hit = False
            if hit:
                counts[name] += 1
                hit_any = True
                if matches is not None:
                    matches[name].append(gi)
        if hit_any:
            any_count += 1
        v = rec.get(default_field)
        lengths.append(len(v) if isinstance(v, str) else 0)
        n_eval += 1
        if n_eval % 500 == 0:
            dx.report_progress('evaluate filter', n_eval, n_total,
                               note=f"{any_count:,} flagged")

    per_10k = lambda c: c / max(n_eval, 1) * 10_000
    result = {
        'filter': fdef['name'], 'version': fdef.get('version', 1),
        'evaluated': n_eval, 'total': total,
        'sampled': bool(sample and total > (sample or 0)),
        'median_chars': int(np.median(lengths)) if lengths else 0,
        'rules': [{'name': name, 'count': counts[name],
                   'per_10k': round(per_10k(counts[name]), 1),
                   'pct': round(counts[name] / max(n_eval, 1) * 100, 2)}
                  for name, _ in compiled],
        'any': {'count': any_count,
                'per_10k': round(per_10k(any_count), 1),
                'pct': round(any_count / max(n_eval, 1) * 100, 2)},
        'sets': [],
    }
    for r in result['rules']:
        print(f"  {r['name']:<20} {r['count']:>8,}  {r['per_10k']:>8.1f}/10k  {r['pct']:>6.2f}%")
    print(f"  {'ANY':<20} {result['any']['count']:>8,}  "
          f"{result['any']['per_10k']:>8.1f}/10k  {result['any']['pct']:>6.2f}%")

    if matches is not None:
        # Sets are stamped with version AND a rules-only hash: a scrub edit
        # bumps the version but leaves r= intact, so drop sets stay fresh.
        stamp = f"v{fdef.get('version', 1)} r={_rules_hash(fdef)}"
        all_idx = sorted(set().union(*matches.values())) if matches else []
        for name, idx in matches.items():
            if idx:
                sname = explorer.store_result_set(
                    idx, f"filter {fdef['name']}.{name} {stamp}",
                    'filter', name=f"{fdef['name']}.{name}", activate=False)
                result['sets'].append(sname)
        if all_idx:
            sname = explorer.store_result_set(
                all_idx, f"filter {fdef['name']} (any rule) {stamp}",
                'filter', name=f"{fdef['name']}.any", activate=False)
            result['sets'].append(sname)
        print(f"  materialized {len(result['sets'])} set(s)")

    # record the observation for the side-by-side comparison view
    if REGISTRY is not None and fdef.get('id'):
        try:
            parent = REGISTRY.find_by_path(explorer.original_filepath)
            REGISTRY.record_filter_eval(fdef['id'], {
                'dataset': (parent['name'] if parent
                            else explorer.original_filepath.name),
                'path': str(explorer.original_filepath.absolute()),
                'version': result['version'],
                'sample': (sample if result['sampled'] else None),
                'evaluated': result['evaluated'], 'total': result['total'],
                'median_chars': result['median_chars'],
                'rules': result['rules'], 'any': result['any'],
                'materialized': materialize,
                'created': time.time(),
            })
        except Exception as e:
            print(f"(eval history not recorded: {e})")
    return result


# ---- tokenize (pre_tokenize.py integration) -------------------------------

class TokenizeParams(BaseModel):
    """Mirror of pre_tokenize.py's CLI, structured. Stored verbatim as the
    linkage recipe on the derived dataset, so any tokenization is reproducible
    and clonable."""
    out_dir: str
    input_format: str = 'auto'      # auto|json|jsonl|parquet|scanned-book-jsonl|batch
    field: Optional[str] = 'text'
    template: Optional[str] = None  # overrides field (NestedFormatter syntax)
    tokenizer: str                  # llama|hf|tiktoken|claude — REQUIRED, no default
    tokenizer_path: Optional[str] = None
    shard_size: int = 100_000_000
    val_holdout: int = 50_000_000
    coprime: int = 6
    min_shard: int = 5_000_000
    dtype: str = 'auto'
    label: str = 'data'
    workers: Optional[int] = None
    legacy_river: bool = False
    batch_size: int = 1000
    # language filtering
    filter_english: bool = False
    lang_threshold: float = 0.8
    lang_backend: str = 'auto'
    lang_model: Optional[str] = None
    lang_sample_size: int = 500
    # scanned-book-jsonl page thresholds
    max_non_dict_ratio: float = 0.50
    min_alpha_ratio: float = 0.60
    min_char_count: int = 150
    max_repetition_ratio: float = 0.05
    include_matter: bool = False
    extra_args: Optional[str] = None    # escape hatch, shlex-split verbatim
    register_as: Optional[str] = None
    # Set by the composed-tokenize job (stream mode), never by the UI directly:
    # pre_tokenize reads the source through this view (file list + skip
    # ordinals) instead of a materialized intermediate.
    view_manifest: Optional[str] = None


class TokenizePreviewRequest(BaseModel):
    field: Optional[str] = None
    template: Optional[str] = None
    n: int = 3


import shlex
import string as _string
import subprocess


class _NestedFormatter(_string.Formatter):
    """pre_tokenize.py's NestedFormatter, replicated for preview: dotted paths
    ({meta.title}) and quoted keys ({rec['some key']})."""
    _pat = re.compile(r"(?:\.|^)(?P<name>[a-zA-Z_]\w*)|\[(?P<q>['\"])(?P<key>.+?)(?P=q)\]")

    def get_field(self, field_name, args, kwargs):
        obj = kwargs
        for m in self._pat.finditer(field_name):
            part = m.group('name') or m.group('key')
            try:
                obj = obj[part]
            except (KeyError, TypeError) as ke:
                keys = ', '.join(f"'{k}'" for k in obj.keys()) \
                    if hasattr(obj, 'keys') else type(obj).__name__
                raise KeyError(f"Key '{part}' not found; available: {keys}") from ke
        return obj, field_name


_TOKFMT = _NestedFormatter()


def _tokenize_argv(source_path: str, p: TokenizeParams) -> List[str]:
    """The exact pre_tokenize.py invocation for these params. One builder used
    by preflight, the runner, and the stored recipe -- they can never drift."""
    argv = [sys.executable, '-u', str(TOOLS_DIR / 'pre_tokenize.py'),
            source_path, str(Path(p.out_dir).expanduser()),
            '--input-format', p.input_format,
            '--tokenizer', p.tokenizer,
            '--shard-size', str(p.shard_size),
            '--val-holdout', str(p.val_holdout),
            '--coprime', str(p.coprime),
            '--min-shard', str(p.min_shard),
            '--dtype', p.dtype,
            '--label', p.label,
            '--batch-size', str(p.batch_size)]
    if p.template:
        argv += ['--format', p.template]
    elif p.field:
        argv += ['--field', p.field]
    if p.tokenizer_path:
        argv += ['--tokenizer_path', p.tokenizer_path]
    if p.view_manifest:
        argv += ['--view-manifest', p.view_manifest]
    if p.workers:
        argv += ['--workers', str(p.workers)]
    if p.legacy_river:
        argv += ['--legacy-river']
    if p.filter_english:
        argv += ['--filter-english', '--lang-threshold', str(p.lang_threshold),
                 '--lang-backend', p.lang_backend,
                 '--lang-sample-size', str(p.lang_sample_size)]
        if p.lang_model:
            argv += ['--lang-model', p.lang_model]
    if p.input_format in ('scanned-book-jsonl', 'batch'):
        argv += ['--max-non-dict-ratio', str(p.max_non_dict_ratio),
                 '--min-alpha-ratio', str(p.min_alpha_ratio),
                 '--min-char-count', str(p.min_char_count),
                 '--max-repetition-ratio', str(p.max_repetition_ratio)]
        if p.include_matter:
            argv += ['--include-matter']
    if p.extra_args:
        argv += shlex.split(p.extra_args)
    return argv


def _tokenize_input_files(source: Path, fmt: str) -> List[str]:
    """Replicates pre_tokenize.py's file-gathering, for preflight counts."""
    import glob as _glob
    if source.is_file():
        return [str(source)]
    pats = []
    if fmt in ('auto', 'parquet'):
        pats.append('**/*.parquet')
    if fmt in ('auto', 'jsonl', 'scanned-book-jsonl'):
        pats += ['**/*.jsonl', '**/*.jsonl.zst']
    if fmt in ('auto', 'json'):
        pats.append('**/*.json')
    files: List[str] = []
    for pat in pats:
        files += _glob.glob(str(source / pat), recursive=True)
    return files


@app.post("/api/datasets/{dataset_id}/tokenize/preview")
def tokenize_preview(dataset_id: str, req: TokenizePreviewRequest):
    """Render the extraction template against REAL records, synchronously.
    Surfaces template KeyErrors before launch instead of as [skip] spam at
    hour two of a run."""
    explorer = _explorer(dataset_id)
    template = req.template or ('{%s}' % (req.field or 'text'))
    total = int(explorer.metadata.get('num_rows') or 0)
    picks = sorted({0, total // 2, max(total - 1, 0)})[:max(1, min(req.n, 10))]
    out = []
    for gi in picks:
        try:
            rec = _get_record_dict(explorer, gi)
            txt = _TOKFMT.format(template, **rec).strip()
            out.append({'index': gi, 'ok': True,
                        'chars': len(txt), 'preview': txt[:400]})
        except KeyError as e:
            out.append({'index': gi, 'ok': False, 'error': str(e)})
        except Exception as e:
            out.append({'index': gi, 'ok': False, 'error': f"{type(e).__name__}: {e}"})
    return {'template': template, 'samples': out}


@app.post("/api/datasets/{dataset_id}/tokenize/preflight")
def tokenize_preflight(dataset_id: str, p: TokenizeParams):
    """Everything knowable BEFORE committing hours: tokenizer loads (vocab ->
    dtype), input files found, template renders, output/resume state,
    size estimates. Runs as a job (tokenizer load can take a while)."""
    entry = _entry(dataset_id)
    explorer = _explorer(dataset_id)
    source = Path(entry.path)

    def _preflight():
        report: Dict[str, Any] = {'ok': True, 'checks': []}

        def check(name, ok, detail):
            report['checks'].append({'name': name, 'ok': bool(ok), 'detail': detail})
            if not ok:
                report['ok'] = False
            print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")

        # tokenizer
        try:
            tok = dx.load_npy_tokenizer(p.tokenizer, p.tokenizer_path or '')
            vocab = len(tok)
            dtype = p.dtype if p.dtype != 'auto' else (
                'uint16' if vocab < 65_536 else 'uint32')
            if p.dtype == 'uint16' and vocab >= 65_536:
                check('tokenizer', False,
                      f"vocab {vocab:,} does not fit uint16 -- use auto/uint32")
            else:
                check('tokenizer', True,
                      f"{p.tokenizer} loads: vocab {vocab:,}, bos {tok.bos_id} "
                      f"-> {dtype} shards")
            report['vocab_size'] = vocab
            report['resolved_dtype'] = dtype
        except Exception as e:
            check('tokenizer', False, f"failed to load: {e}")

        # input files
        files = _tokenize_input_files(source, p.input_format)
        check('input files', bool(files),
              f"{len(files)} file(s) match format {p.input_format!r}")
        report['n_input_files'] = len(files)

        # template render on real records
        prev = tokenize_preview(dataset_id, TokenizePreviewRequest(
            field=p.field, template=p.template, n=3))
        bad = [s for s in prev['samples'] if not s['ok']]
        check('extraction', not bad,
              (f"template renders on {len(prev['samples'])} sample records"
               if not bad else f"template fails: {bad[0]['error']}"))
        report['samples'] = prev['samples']

        # output dir / resume state
        out = Path(p.out_dir).expanduser()
        manifest = out / f"{p.label}.manifest"
        if manifest.exists():
            done_files, prev_tokens = 0, 0
            for line in manifest.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    done_files += 1
                    if '\t' in line:
                        prev_tokens += int(line.rsplit('\t', 1)[1])
            check('output', True,
                  f"RESUME: manifest found -- {done_files} file(s) already done "
                  f"({prev_tokens:,} tokens); this run continues where it left off")
            report['resume'] = {'files_done': done_files, 'tokens': prev_tokens}
        elif out.exists() and any(out.iterdir()):
            check('output', True,
                  f"{out} exists and is non-empty (no {p.label}.manifest -- "
                  f"shards will APPEND after the highest existing index)")
        else:
            check('output', True, f"fresh output directory {out}")

        # estimates from known dataset size (~4 chars/token heuristic)
        size_mb = explorer.metadata.get('file_size') or 0
        if size_mb:
            est_tokens = int(size_mb * 1e6 / 4)
            report['est_tokens'] = est_tokens
            report['est_shards'] = max(1, est_tokens // max(p.shard_size, 1))
            check('estimate', True,
                  f"~{est_tokens / 1e9:.2f}B tokens -> ~{report['est_shards']} "
                  f"shard(s) of {p.shard_size:,} (rough 4 chars/token)")

        report['argv'] = _tokenize_argv(str(source), p)
        print('  command: ' + ' '.join(shlex.quote(a) for a in report['argv'][2:]))
        return report

    job = entry.worker.submit('tokenize-preflight', _preflight,
                              {'params': p.model_dump()})
    return {'job_id': job.id}


_TOKPROG_RE = re.compile(
    r"===\s*(\d+)/(\d+)\s+files\s*\(([\d.]+)%\)\s*•\s*([\d,]+)\s+tokens")


def _make_tokenize_runner(source: Path, p: TokenizeParams, total_docs: int,
                          extra_recipe: Optional[Dict[str, Any]] = None,
                          lineage_path: Optional[Path] = None):
    """Build the tokenize job body: pre_tokenize.py as a watched subprocess
    with parsed progress, then library registration. Module-level factory so
    the composed-tokenize job can chain it after materialize+export on the
    same worker thread. `lineage_path` overrides which path the result's
    derived_from is resolved against (an ephemeral intermediate that will be
    deleted should not be anyone's parent)."""
    out = Path(p.out_dir).expanduser()

    def _run():
        argv = _tokenize_argv(str(source), p)
        print('launching: ' + ' '.join(shlex.quote(a) for a in argv[2:]))
        out.mkdir(parents=True, exist_ok=True)
        # cwd MUST be the tools dir: pre_tokenize sys.path-inserts
        # '../common_fsdp2' relative to its working directory.
        # PYTHONUTF8=1: on Windows a piped stdout defaults to cp1252, and any
        # character outside it crashes the CHILD's print -- fatally, at the end
        # of an arbitrarily long run. Force UTF-8 both ways; decode leniently.
        env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
        proc = subprocess.Popen(argv, cwd=str(TOOLS_DIR), env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True, bufsize=1,
                                encoding='utf-8', errors='replace',
                                start_new_session=(os.name != 'nt'))
        this_job = getattr(_current_job, 'job', None)

        # Quiet-child watchdog: worker spawn and shard-scan phases can be
        # SILENT for minutes (everything loads over a network share). Surface
        # "alive but quiet" into the progress line instead of dead air --
        # by RE-EMITTING the last known docs progress with a quiet note
        # appended, so pct/ETA survive the silence instead of being clobbered
        # by a 0/0 note-only report.
        last_out = [time.time()]
        stop_wd = threading.Event()
        quiet_note = ['']

        def _watchdog():
            _current_job.job = this_job      # thread-local: re-bind in this thread
            try:
                while not stop_wd.wait(5.0):
                    if this_job is not None and this_job.cancel_requested:
                        print('[cancel] killing pre_tokenize process tree...')
                        _kill_proc_tree(proc)
                        return
                    quiet = time.time() - last_out[0]
                    if quiet >= 10:
                        quiet_note[0] = (f" · subprocess quiet {int(quiet)}s "
                                         f"(worker spawn / shard scans are "
                                         f"silent phases)")
                        _report_docs()
            except BaseException:
                pass                          # cancellation raise: just stop

        wd = threading.Thread(target=_watchdog, daemon=True)
        wd.start()
        # Four progress signals, because '=== N/M files ===' alone is useless
        # for the common single-huge-file dataset (first event at 100%):
        #   [<file>] queued N docs      -- feed heartbeat, every 50k docs
        #   [write] shard (N tokens, M docs)  -- durable progress per shard
        #   === N/M files ===           -- per-file completion (main level)
        #   Scanning <label>: i/n shards -- the token-cache pass (\r lines!)
        queue_re = re.compile(r"^\[(.+?)\] queued ([\d,]+) docs")
        scan_re = re.compile(r"Scanning\s+\S+:\s*(\d+)/(\d+)\s+shards")
        write_re = re.compile(r"^\[write\] .*\(([\d,]+) tokens(?:, ([\d,]+) docs)?\)")
        scrub_re = re.compile(r"^\[scrub\] (\S+): docs=([\d,]+) subs=([\d,]+) "
                              r"chars_removed=(-?[\d,]+)")
        scrub_nc_re = re.compile(r"^\[scrub\] fixpoint nonconverged=([\d,]+)")
        scrub_counts: Dict[str, Any] = {}
        n_tokens = 0
        docs_written = 0
        queued = 0
        file_base_q = 0
        cur_file = None

        def _report_docs():
            done = docs_written if docs_written else file_base_q + queued
            note = f"{n_tokens:,} tokens written · {file_base_q + queued:,} docs queued"
            if quiet_note[0]:
                note += quiet_note[0]
            dx.report_progress('tokenize docs',
                               min(done, total_docs) if total_docs else done,
                               total_docs, note=note)

        def _iter_segments(stream):
            """Yield (text, is_cr) split on BOTH \\n and \\r. pre_tokenize's
            shard-scan phase prints \\r-rewriting lines with no newline for
            minutes -- a plain line iterator shows NOTHING until it ends."""
            buf = ''
            while True:
                ch = stream.read(1)
                if ch == '':
                    if buf:
                        yield buf, False
                    return
                if ch == '\n':
                    yield buf, False
                    buf = ''
                elif ch == '\r':
                    if buf:
                        yield buf, True
                    buf = ''
                else:
                    buf += ch

        try:
            for line, is_cr in _iter_segments(proc.stdout):
                if this_job is not None and this_job.cancel_requested:
                    print('[cancel] killing pre_tokenize process tree...')
                    _kill_proc_tree(proc)
                    proc.wait()
                    raise JobCancelled()
                last_out[0] = time.time()
                quiet_note[0] = ''            # child spoke: drop the quiet note
                # Reproduce \r semantics into the job log (its writer collapses
                # rewrites into one updating line).
                print(line, end='\r' if is_cr else '\n', flush=True)
                m = scan_re.search(line)
                if m:
                    dx.report_progress('scan output shards', int(m.group(1)),
                                       int(m.group(2)))
                    continue
                m = queue_re.match(line)
                if m:
                    if m.group(1) != cur_file:
                        file_base_q += queued
                        queued = 0
                        cur_file = m.group(1)
                    queued = int(m.group(2).replace(',', ''))
                    _report_docs()
                    continue
                m = write_re.match(line)
                if m:
                    n_tokens += int(m.group(1).replace(',', ''))
                    if m.group(2):
                        docs_written += int(m.group(2).replace(',', ''))
                    _report_docs()
                    continue
                m = scrub_re.match(line)
                if m:
                    if m.group(1) == 'fixpoint':    # "[scrub] fixpoint nonconverged=N"
                        pass                        # falls through to the nc parse below
                    else:
                        scrub_counts[m.group(1)] = {
                            'docs': int(m.group(2).replace(',', '')),
                            'subs': int(m.group(3).replace(',', '')),
                            'chars_removed': int(m.group(4).replace(',', ''))}
                        continue
                m = scrub_nc_re.match(line)
                if m:
                    scrub_counts['__nonconverged__'] = int(m.group(1).replace(',', ''))
                    continue
                m = _TOKPROG_RE.search(line)
                if m:
                    dx.report_progress('tokenize files', int(m.group(1)),
                                       int(m.group(2)), main=True,
                                       note=f"{m.group(4)} tokens")
        finally:
            stop_wd.set()                 # watchdog dies on EVERY exit path
        rc = proc.wait()
        if this_job is not None and this_job.cancel_requested:
            raise JobCancelled()
        if rc != 0:
            raise RuntimeError(f"pre_tokenize.py exited with code {rc}")

        # stats straight from the shard manifests DocShardWriter wrote
        stats = _quick_stats(out)
        stats.pop('kind_guess', None)
        result = {'out_dir': str(out), 'tokens': stats.get('tokens') or n_tokens,
                  'num_rows': stats.get('num_rows'),
                  'num_files': stats.get('num_files'), 'registered': None}
        if scrub_counts:
            result['scrub_counts'] = scrub_counts

        if REGISTRY is not None and p.register_as:
            try:
                parent = REGISTRY.find_by_path(lineage_path or source)
                rid = REGISTRY.unique_id(p.register_as)
                recipe = {'kind': 'tokenize',
                          'params': p.model_dump(),
                          'argv': _tokenize_argv(str(source), p)[2:],
                          'created': time.time()}
                if extra_recipe:
                    recipe.update(extra_recipe)
                REGISTRY.upsert({
                    'id': rid,
                    'name': p.register_as,
                    'path': str(out.absolute()),
                    'kind': 'tokenized',
                    'tags': ['tokenized', p.tokenizer],
                    'notes': f"tokenized from {source.name} "
                             f"({p.tokenizer}, label {p.label!r})",
                    'derived_from': parent['id'] if parent else None,
                    # The LINKAGE carries the full recipe: reproducible,
                    # clonable, and comparable across sibling tokenizations.
                    'recipe': recipe,
                    'open_opts': {'text_field': None,
                                  'tok_kind': p.tokenizer,
                                  'tok_path': p.tokenizer_path,
                                  'special_tokens': None,
                                  'raw_shards': True, 'recursive': False},
                    'stats': _jsonable(stats),
                    'created': time.time(),
                    'last_opened': None,
                })
                result['registered'] = rid
                print(f"registered in library as {p.register_as!r}"
                      + (f" (derived from {parent['name']})" if parent else ""))
            except Exception as e:
                print(f"(library registration failed: {e})")
        return result

    return _run


@app.post("/api/datasets/{dataset_id}/tokenize")
def tokenize(dataset_id: str, p: TokenizeParams):
    entry = _entry(dataset_id)
    explorer = _explorer(dataset_id)
    source = Path(entry.path)
    if not p.tokenizer:
        raise HTTPException(400, "tokenizer is required")
    out = Path(p.out_dir).expanduser()
    try:
        if source.is_dir() and out.resolve() == source.resolve():
            raise HTTPException(400, "Output must differ from the source directory")
    except OSError:
        pass
    job = entry.worker.submit(
        'tokenize',
        _make_tokenize_runner(source, p,
                              int(explorer.metadata.get('num_rows') or 0)),
        {'params': p.model_dump()})
    return {'job_id': job.id}


class ComposedTokenizeRequest(BaseModel):
    """Tokenize a COMPOSED view of a dataset: (union of include sets, or all
    records) minus exclude sets minus the filter's ANY-rule matches.

    Two intermediate modes -- the materialization-tier choice:
    - 'stream' (pointer tier): write a tiny view manifest (file list + per-file
      skip ordinals) into the output dir and let pre_tokenize read the SOURCE
      through it, dropping excluded records in-stream. No byte copy at all --
      the right default when the composition only selects records.
    - 'file' (full tier): export a plain-JSONL `*_cleaned` intermediate
      (registered, browsable, carries lineage) and tokenize THAT. For when the
      cleaned corpus is itself a wanted artifact."""
    include_sets: List[str] = []
    exclude_sets: List[str] = []
    filter_id: Optional[str] = None       # drops (predicate -> sets)
    transform_id: Optional[str] = None    # rewrites (function chain on survivors)
    intermediate_mode: str = 'file'    # 'stream' | 'file'
    intermediate_path: Optional[str] = None       # file mode only
    intermediate_register_as: Optional[str] = None
    keep_intermediate: bool = True     # file mode: False deletes the file after
                                       # a successful tokenize (ephemeral grade)
    tokenize: TokenizeParams


@app.post("/api/datasets/{dataset_id}/tokenize/composed")
def tokenize_composed(dataset_id: str, req: ComposedTokenizeRequest):
    entry = _entry(dataset_id)
    explorer = _explorer(dataset_id)
    source = Path(entry.path)
    p = req.tokenize
    if not p.tokenizer:
        raise HTTPException(400, "tokenizer is required")
    if not (req.include_sets or req.exclude_sets or req.filter_id):
        raise HTTPException(400, "Empty composition -- use the plain "
                                 "tokenize endpoint instead")
    for name in req.include_sets + req.exclude_sets:
        _set_indices(explorer, name)               # validate before queueing
    fdef = _registry().get_filter(req.filter_id) if req.filter_id else None
    tdef = _registry().get_transform(req.transform_id) if req.transform_id else None
    if tdef:
        _compile_scrubs(tdef['scrubs'], 'text')    # fail fast, not mid-job
    if req.intermediate_mode not in ('stream', 'file'):
        raise HTTPException(400, f"Unknown intermediate_mode {req.intermediate_mode!r}")
    ipath = None
    if req.intermediate_mode == 'file':
        if not req.intermediate_path:
            raise HTTPException(400, "intermediate_path is required in file mode")
        ipath = Path(req.intermediate_path).expanduser()
        if ipath.exists():
            raise HTTPException(409, "Refusing to overwrite existing "
                                     f"intermediate: {ipath}")
    elif explorer.file_type not in ('jsonl', 'parquet'):
        raise HTTPException(400, "stream mode needs a jsonl/parquet source whose "
                                 f"record order pre_tokenize can reproduce (this is "
                                 f"{explorer.file_type!r}) -- use materialize mode")
    try:
        if source.is_dir() and Path(p.out_dir).expanduser().resolve() \
                == source.resolve():
            raise HTTPException(400, "Output must differ from the source directory")
    except OSError:
        pass

    def _run_composed():
        # -- 1/3: filter -> sets (skip if this version's sets already exist)
        exclude = list(req.exclude_sets)
        if fdef is not None:
            aname = f"{fdef['name']}.any"
            ver = fdef.get('version', 1)
            rhash = _rules_hash(fdef)
            cur = explorer.result_sets.get(aname)
            q = str(cur.get('query', '')) if cur is not None else ''
            # Fresh if the RULES match (r= stamp; scrub edits don't invalidate);
            # legacy sets without a stamp fall back to exact-version match.
            fresh = (f"r={rhash}" in q) or q.endswith(f"v{ver}")
            if cur is not None and fresh:
                print(f"[compose 1/3] filter {fdef['name']!r} rules unchanged since "
                      f"materialize ({cur['count']:,} records in {aname!r}) -- reusing")
            else:
                print(f"[compose 1/3] materializing filter {fdef['name']!r} v{ver}")
                _filter_eval_impl(explorer, fdef, sample=None, materialize=True)
            if aname in explorer.result_sets:
                exclude.append(aname)
            else:
                print("  filter matched 0 records -- nothing to exclude")
        else:
            print("[compose 1/3] no filter selected -- skipped")

        keep = _export_keep(explorer, req.include_sets, exclude)
        total = int(explorer.metadata.get('num_rows') or 0)
        if keep.size == 0:
            raise RuntimeError("Composition keeps 0 records -- nothing to tokenize")
        n_kept = int(keep.size)
        n_dropped = total - n_kept
        # Scrubs come from the TRANSFORM: resolve field names NOW so the
        # recipe (and the view manifest) carry concrete fields, not "whatever
        # the default text field happens to be later".
        scrub_defs: List[Dict[str, str]] = []
        if tdef:
            sfield = explorer._resolve_text_field(None)
            for s in tdef['scrubs']:
                sd = ScrubDef(**s)
                # literal/line scrubs resolve to their FINAL regex here, so the
                # view-manifest contract stays "pattern is a regex" and
                # pre_tokenize needs no mode awareness
                scrub_defs.append({'name': sd.name, 'field': sd.field or sfield,
                                   'pattern': _scrub_regex_source(sd),
                                   'replacement': _scrub_repl_source(sd)})
        scrub_counts: Dict[str, List[int]] = {}
        scrub_fixpoint = bool(tdef.get('fixpoint')) if tdef else False
        composition = {
            'source': str(source.absolute()),
            'include_sets': req.include_sets,
            'exclude_sets': req.exclude_sets,
            'filter': ({'id': req.filter_id, 'name': fdef['name'],
                        'version': fdef.get('version', 1)} if fdef else None),
            'transform': ({'id': req.transform_id, 'name': tdef['name'],
                           'version': tdef.get('version', 1)} if tdef else None),
            'mode': req.intermediate_mode,
            'scrubs': scrub_defs,
            'scrub_fixpoint': scrub_fixpoint,
            'kept_records': n_kept,
            'dropped_records': n_dropped,
        }

        if req.intermediate_mode == 'stream':
            # -- 2/3 (pointer tier): view manifest instead of a byte copy.
            # Written into the output dir so the shard set is self-describing.
            out = Path(p.out_dir).expanduser()
            out.mkdir(parents=True, exist_ok=True)
            view_json = out / f"{p.label}.view.json"
            view_npz = out / f"{p.label}.view.npz"
            skips = np.setdiff1d(np.arange(total, dtype=np.int64), keep)
            if explorer.is_directory:
                cum = explorer.cum_record_counts
                srcs = explorer.source_files
            else:
                cum = [0, total]
                srcs = [explorer.original_filepath]
            arrays: Dict[str, np.ndarray] = {}
            files_meta = []
            for fi, srcp in enumerate(srcs):
                lo, hi = int(cum[fi]), int(cum[fi + 1])
                arr = skips[np.searchsorted(skips, lo):
                            np.searchsorted(skips, hi)] - lo
                arrays[f'skip_{fi:05d}'] = arr
                files_meta.append({'path': str(srcp), 'records': hi - lo})
            np.savez_compressed(view_npz, **arrays)
            view_json.write_text(json.dumps({
                'source': str(source.absolute()),
                'files': files_meta,
                'skips': view_npz.name,
                'scrubs': scrub_defs,
                'scrub_fixpoint': scrub_fixpoint,
                'kept': n_kept,
                'dropped': n_dropped,
                'composition': composition,
                'created': time.time(),
            }, indent=1), encoding='utf-8')
            print(f"[compose 2/3] view manifest {view_json.name}: "
                  f"{n_dropped:,} skip ordinal(s) across {len(srcs)} file(s)"
                  + (f", {len(scrub_defs)} scrub(s) applied in-stream"
                     if scrub_defs else "")
                  + f" -- no intermediate copy, {n_kept:,} records stream from source")
            composition['view_manifest'] = str(view_json.absolute())

            # -- 3/3: tokenize the source THROUGH the view
            print(f"[compose 3/3] tokenizing {source.name} (streamed view)")
            runner = _make_tokenize_runner(
                source, p.model_copy(update={'view_manifest': str(view_json)}),
                n_kept, extra_recipe={'composition': composition})
            result = runner()
            result['intermediate'] = None
            result['view_manifest'] = str(view_json)
        else:
            # -- 2/3 (full tier): export the survivors as a cleaned intermediate
            print(f"[compose 2/3] exporting {n_kept:,} of {total:,} records "
                  f"-> {ipath}")
            ipath.parent.mkdir(parents=True, exist_ok=True)
            part = ipath.with_name(ipath.name + '.part')
            compiled_scrubs = _compile_scrubs(scrub_defs, None) if scrub_defs else []

            n_nonconverged = [0]

            def _out_lines():
                for line in _iter_kept_lines(explorer, keep):
                    if not compiled_scrubs:
                        yield line
                        continue
                    rec = json.loads(line)
                    _c, _p, converged, _h = _apply_scrubs(
                        rec, compiled_scrubs, scrub_counts,
                        fixpoint=scrub_fixpoint)
                    if not converged:
                        n_nonconverged[0] += 1
                    yield (json.dumps(rec, ensure_ascii=False) + '\n').encode()

            n_written = 0
            try:
                with open(part, 'wb') as fout:
                    for line in _out_lines():
                        if not line.endswith(b'\n'):
                            line += b'\n'
                        fout.write(line)
                        n_written += 1
                        if n_written % 2000 == 0:
                            dx.report_progress('export intermediate',
                                               n_written, n_kept)
            except BaseException:            # incl. JobCancelled: no .part garbage
                part.unlink(missing_ok=True)
                raise
            part.replace(ipath)
            print(f"  intermediate complete: {n_written:,} records "
                  f"({n_dropped:,} dropped)")
            for name, _f, _p, _r in compiled_scrubs:
                c = scrub_counts.get(name, [0, 0, 0])
                print(f"  [scrub] {name}: docs={c[0]:,} subs={c[1]:,} "
                      f"chars_removed={c[2]:,}")
            if scrub_fixpoint and n_nonconverged[0]:
                print(f"  [scrub] WARNING: {n_nonconverged[0]:,} record(s) did "
                      f"not converge at {SCRUB_MAX_PASSES} passes")
            if compiled_scrubs:
                composition['scrub_counts'] = {
                    n2: {'docs': c[0], 'subs': c[1], 'chars_removed': c[2]}
                    for n2, c in scrub_counts.items()}
            if req.keep_intermediate:
                _register_export(explorer, ipath, ExportRequest(
                    out_path=str(ipath), include=req.include_sets or None,
                    exclude=exclude or None,
                    register_as=req.intermediate_register_as), n_written)

            # -- 3/3: tokenize the intermediate
            print(f"[compose 3/3] tokenizing {ipath.name}")
            composition['intermediate'] = str(ipath.absolute())
            composition['intermediate_kept'] = req.keep_intermediate
            runner = _make_tokenize_runner(
                ipath, p, n_written, extra_recipe={'composition': composition},
                lineage_path=None if req.keep_intermediate else source)
            result = runner()
            result['intermediate'] = str(ipath)
            if composition.get('scrub_counts'):
                result['scrub_counts'] = composition['scrub_counts']
            if not req.keep_intermediate:
                try:
                    ipath.unlink()
                    print(f"deleted ephemeral intermediate {ipath}")
                except OSError as e:
                    print(f"(could not delete intermediate: {e})")

        result['kept_records'] = n_kept
        result['dropped_records'] = n_dropped
        return result

    job = entry.worker.submit('tokenize-composed', _run_composed,
                              {'params': req.model_dump()})
    return {'job_id': job.id}


class MigrateRequest(BaseModel):
    path: str


class ResolveConflictRequest(BaseModel):
    target: str          # the correct new-hash filename (full path)
    keep: str            # chosen candidate (full path) -> renamed to target
    others: List[str] = []   # losing candidates -> sidelined as *.superseded


@app.post("/api/resolve-conflict")
def resolve_conflict(req: ResolveConflictRequest):
    """Apply a human keep/discard decision from the conflict chooser.

    The chosen candidate takes the target name; losers are sidelined (renamed
    *.superseded, never deleted) so a wrong choice is reversible by hand.
    """
    target, keep = Path(req.target), Path(req.keep)
    involved = [target, keep] + [Path(o) for o in req.others]
    for p in involved:
        if p.parent.name not in ('.dataset_explorer_cache', 'tmp') \
                or p.parent != target.parent:
            raise HTTPException(400, f"Refusing to touch {p}: outside the "
                                     f"conflict's cache directory")
    if not keep.exists():
        raise HTTPException(404, f"Chosen candidate missing: {keep}")
    if target.exists():
        raise HTTPException(409, f"Target already exists: {target}")
    for o in req.others:
        op = Path(o)
        if op.exists():
            op.rename(op.with_name(op.name + '.superseded'))
    keep.rename(target)
    return {'kept': str(target),
            'sidelined': [o + '.superseded' for o in req.others]}


@app.post("/api/migrate-cache")
def migrate_cache_endpoint(req: MigrateRequest):
    """Adopt path-hash-keyed caches after a dataset was moved or copied.

    Runs as a JOB: repairing cache timestamps re-gzips the line-position
    indexes, which takes minutes on a large corpus -- a synchronous endpoint
    here just looks like a hung button. Renames only; content is re-validated
    by the normal loaders on open, and re-running is a safe no-op.
    """
    path = Path(req.path).expanduser()
    if not path.exists():
        raise HTTPException(400, f"Path does not exist: {path}")
    job = UTILITY_WORKER.submit(
        'migrate', lambda: dx.migrate_cache(str(path)), {'path': str(path)})
    return {'job_id': job.id}


# ---- jobs -----------------------------------------------------------------

@app.get("/api/jobs")
def list_jobs(limit: int = 50):
    jobs = sorted(JOBS.values(), key=lambda j: -j.created)[:limit]
    return [{'id': j.id, 'kind': j.kind, 'dataset_id': j.dataset_id,
             'status': j.status, 'created': j.created, 'finished': j.finished,
             'cancel_requested': j.cancel_requested,
             'error': j.error, 'progress': _jsonable(j.progress)} for j in jobs]


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"No job {job_id!r}")
    if job.status in ('done', 'error', 'cancelled'):
        return {'status': job.status}
    job.cancel_requested = True
    if job.status == 'queued':
        job.status = 'cancelled'
        job.finished = time.time()
        job.log.append('[cancelled while queued]')
    return {'status': 'cancelling'}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, log_from: int = 0):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"No job {job_id!r}")
    return job.summary(log_from=log_from)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"No job {job_id!r}")

    import asyncio

    async def _stream():
        # Lines are BATCHED per poll tick (capped per event). A directory load
        # can print tens of thousands of lines; one event per line melts the
        # browser's event loop long before bandwidth matters.
        sent = 0
        last_status = None
        last_progress = None
        last_sent_text = None
        while True:
            if job.status != last_status:
                last_status = job.status
                yield f"event: status\ndata: {json.dumps({'status': job.status})}\n\n"
            prog = job.progress
            if prog and prog.get('updated') != last_progress:
                last_progress = prog.get('updated')
                yield f"event: progress\ndata: {json.dumps(_jsonable(prog))}\n\n"
            # A \r-rewritten line MUTATES log[-1] after it was already sent;
            # without this, live viewers never see progress-style rewrites.
            if sent and sent == len(job.log) and job.log \
                    and job.log[-1] != last_sent_text:
                last_sent_text = job.log[-1]
                yield f"event: replace\ndata: {json.dumps({'line': last_sent_text})}\n\n"
            while sent < len(job.log):
                chunk = job.log[sent:sent + 1000]
                sent += len(chunk)
                last_sent_text = chunk[-1]
                yield f"data: {json.dumps({'lines': chunk})}\n\n"
            if job.status in ('done', 'error', 'cancelled'):
                payload = json.dumps(job.summary(log_from=len(job.log)))
                yield f"event: end\ndata: {payload}\n\n"
                return
            await asyncio.sleep(0.25)

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ---- filesystem browse (for the load form) --------------------------------

DRIVES = '::drives'   # virtual root above the drive letters (Windows only)


def _list_drives() -> List[str]:
    if hasattr(os, 'listdrives'):          # Python 3.12+
        return os.listdrives()
    return [f"{c}:\\" for c in string.ascii_uppercase if Path(f"{c}:\\").exists()]


@app.get("/api/browse")
def browse(path: str = "~"):
    """Non-recursive listing of one directory, for the dataset-picker UI.

    On Windows a virtual '::drives' level sits above each drive root, since
    Path('C:/').parent is itself and mounted drives are unreachable otherwise.
    """
    if os.name == 'nt' and path in (DRIVES, '', '/'):
        return {'path': DRIVES, 'parent': DRIVES,
                'entries': [{'name': d, 'path': d, 'dir': True}
                            for d in _list_drives()]}
    p = Path(path).expanduser()
    if not p.exists():
        raise HTTPException(404, f"Not found: {p}")
    if p.is_file():
        p = p.parent
    # os.scandir, NOT iterdir + per-entry Path.is_dir(): scandir returns each
    # entry's type WITH the directory listing (free on Windows), where the
    # Path approach costs a network stat PER FILE -- minutes on a 6,000-file
    # SMB folder for what is one enumeration's worth of information.
    entries = []
    truncated = False
    try:
        with os.scandir(p) as it:
            children = sorted(it, key=lambda e: e.name.lower())
        for de in children:
            name = de.name
            if name.startswith('.'):
                continue
            try:
                is_dir = de.is_dir()
            except OSError:
                continue
            lname = name.lower()
            is_data = not is_dir and (
                lname.rsplit('.', 1)[-1] in ('parquet', 'jsonl', 'json', 'npy')
                or lname.endswith('.jsonl.zst'))
            if is_dir or is_data:
                if len(entries) >= 500:
                    truncated = True
                    break
                entries.append({'name': name, 'path': str(p / name),
                                'dir': is_dir})
    except PermissionError:
        raise HTTPException(403, f"Permission denied: {p}")
    parent = str(p.parent)
    if os.name == 'nt' and p.parent == p:   # drive root: ".." goes to the drive list
        parent = DRIVES
    return {'path': str(p), 'parent': parent, 'entries': entries,
            'truncated': truncated}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    global REGISTRY, TOKENIZED_ROOT
    ap = argparse.ArgumentParser(description="Dataset Explorer web server")
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--tokenized-root', default=None,
                    help='Root dir for suggested tokenize output paths '
                         '(<root>/<tokenizer>/<dataset>); overrides the '
                         'source->tokenized path heuristic')
    ap.add_argument('--registry', default='~/whynot/traindata/registry.json',
                    help='Path to the managed-dataset registry JSON '
                         '(shareable across rigs, e.g. on a NAS mount)')
    args = ap.parse_args()
    REGISTRY = Registry(Path(args.registry))
    TOKENIZED_ROOT = args.tokenized_root
    print(f"Dataset Explorer Web on http://{args.host}:{args.port}")
    print(f"Registry: {REGISTRY.path}")
    try:
        moved = REGISTRY.migrate_filter_scrubs()
        for fname, tname in moved:
            print(f"  migrated scrubs: filter {fname!r} -> transform {tname!r}")
    except Exception as e:
        print(f"  (filter-scrub migration skipped: {e})")
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
