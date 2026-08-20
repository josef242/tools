#!/usr/bin/env python3
"""
Dataset Explorer - A flashlight for large LLM training dataset files
Supports: Parquet, JSONL, and compressed JSONL.zst files with metadata caching
Author: Claude (Enhanced version with zstandard support)

Key changes from original:
1. Added support for .jsonl.zst files with automatic decompression
2. Uses local 'tmp' directory for cross-platform compatibility
3. All original functionality preserved
"""

import bisect
import json
import shlex
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Sequence, Union, Tuple
import argparse
from collections import Counter
import re
import textwrap
import time
import hashlib
import pickle
import gzip
import tempfile
import atexit
import shutil

try:
    import pandas as pd
    import pyarrow.parquet as pq
    import numpy as np
except ImportError:
    print("Please install required packages:")
    print("pip install pandas pyarrow numpy")
    sys.exit(1)

# Try to import zstandard for .zst support
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better formatting: pip install rich")

# Try to import tqdm for progress bars as a fallback
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# 'findall [-n <name>] -m <query>'. The metadata query is taken verbatim after this
# match, because it contains quotes, parens and spaces that shlex would mangle.
_META_FINDALL_RE = re.compile(
    r'findall\s+(?:-n\s+(?P<name>\S+)\s+)?(?:-m|--meta)\s+', re.I)

# Near-duplicate detection lives in a sibling module so it can be unit-tested without
# opening a dataset. Optional: the explorer works fully without it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import neardupe
    NEARDUPE_AVAILABLE = True
except ImportError:
    NEARDUPE_AVAILABLE = False

try:
    import metaquery
    import metaindex
    METAQUERY_AVAILABLE = True
except ImportError:
    METAQUERY_AVAILABLE = False

# Embedder progress channel. A host application (the web server) installs a callable
# here; the CLI leaves it None, making every report_progress call a no-op. Rich bars
# remain the CLI's display -- this is the machine-readable side channel, throttled and
# formatted by the host. Signature: hook(stage, done, total, main) where `main` marks
# the outer of two nested levels (file 12/400 vs. progress within that file).
PROGRESS_HOOK = None


def report_progress(stage: str, done, total=None, main: bool = False,
                    note: Optional[str] = None, unit: Optional[str] = None):
    """unit='bytes' marks stages whose counters are byte offsets, so displays
    can render '114.3 GB / 120.1 GB' instead of a hallucinatory item count."""
    hook = PROGRESS_HOOK
    if hook is None:
        return
    try:
        hook(str(stage), int(done), int(total or 0), bool(main),
             str(note) if note is not None else None, unit)
    except Exception:
        pass  # progress must never break the work it reports on


def _parse_list_args(args: List[str]) -> Optional[Tuple[Optional[int], int]]:
    """Parse args for the 'list' command. Returns (count, width) or None on error.

    count=None means 'all'. Defaults: count=200, width=100.
    Accepted forms: '', '<n>', 'all', '<n> <width>', 'all <width>'.
    """
    count: Optional[int] = 200
    width = 100

    if args:
        first = args[0]
        if first.lower() == 'all':
            count = None
        else:
            try:
                count = int(first)
            except ValueError:
                print(f"Invalid count: {first} (expected integer or 'all')")
                return None
            if count <= 0:
                print("Count must be a positive integer (or 'all').")
                return None

    if len(args) > 1:
        try:
            width = int(args[1])
        except ValueError:
            print(f"Invalid width: {args[1]}")
            return None
        if width <= 0:
            print("Width must be a positive integer.")
            return None

    if len(args) > 2:
        print("Usage: list [<n>|all] [<width>]")
        return None

    return count, width


def dataset_cache_root(base: Path) -> Path:
    """THE cache directory for a dataset: <dir>/.dataset_explorer_cache for a
    directory dataset, <parent>/.dataset_explorer_cache for a single file.
    Temps (decompressions, conversions, decodes) live in a tmp/ subdirectory of
    it, so a dataset's entire derived state is one tidy, movable package."""
    base = Path(base)
    return (base if base.is_dir() else base.parent) / '.dataset_explorer_cache'


def _stem_variants(src: Path) -> List[str]:
    """Stems an artifact of this source file may carry (compound-ext handling)."""
    out = [src.stem]
    if src.stem.endswith('.jsonl'):
        out.append(src.stem[:-6])
    return out


def consolidate_cache_layout(base: Path, sources: Sequence[Path]) -> int:
    """Move a dataset's cache/temp artifacts from the LEGACY scattered layout
    (per-source-dir .dataset_explorer_cache/ and tmp/) into the consolidated
    root cache. Renames only -- artifact names embed path hashes that do not
    change, so everything still validates after the move. Returns files moved.
    """
    cache_root = dataset_cache_root(base)
    temp_root = cache_root / 'tmp'
    moved = 0
    legacy_dirs = set()
    for i, src in enumerate(sources):
        if i % 200 == 0:
            report_progress('check cache layout', i, len(sources))
        pats = [re.compile(rf"^{re.escape(s)}_[0-9a-f]{{8}}\S*$")
                for s in _stem_variants(src)]

        def _sweep(legacy: Path, target: Path):
            nonlocal moved
            if not legacy.is_dir() or legacy == target:
                return
            legacy_dirs.add(legacy)
            for p in legacy.iterdir():
                if not p.is_file() or p.name.endswith('.part'):
                    continue
                if any(pat.match(p.name) for pat in pats):
                    target.mkdir(parents=True, exist_ok=True)
                    dest = target / p.name
                    if not dest.exists():
                        p.rename(dest)
                        moved += 1

        _sweep(src.parent / '.dataset_explorer_cache', cache_root)
        _sweep(src.parent / 'tmp', temp_root)
    for d in legacy_dirs:       # remove now-empty legacy dirs, never non-empty ones
        try:
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass
    if moved:
        print(f"Consolidated {moved} cache file(s) into {cache_root}")
    return moved


def read_pathkey(path) -> Optional[str]:
    """The path spelling this DIRECTORY dataset's caches are keyed for, if recorded.

    Directory datasets only: a single-file dataset shares its cache dir with
    siblings, so one marker file cannot speak for it. The marker is written on
    every successful open and by migrate_cache, so an embedder can compare it to
    the spelling being opened and adopt BEFORE re-deriving anything.
    """
    base = Path(path).expanduser().absolute()
    if not base.is_dir():
        return None
    marker = base / '.dataset_explorer_cache' / 'pathkey.json'
    try:
        if marker.exists():
            return json.loads(marker.read_text(encoding='utf-8')).get('path')
    except Exception:
        pass
    return None


def write_pathkey(path):
    """Record the spelling this directory dataset's caches are keyed for."""
    base = Path(path).expanduser().absolute()
    if not base.is_dir():
        return
    try:
        cache_dir = base / '.dataset_explorer_cache'
        cache_dir.mkdir(exist_ok=True)
        (cache_dir / 'pathkey.json').write_text(
            json.dumps({'path': str(base), 'updated': time.time()}),
            encoding='utf-8')
    except Exception:
        pass    # marker is advisory; never let it break an open


def needs_adoption(path) -> bool:
    """True if opening `path` as spelled would miss existing caches.

    Cheap by design (a marker read, or one directory listing) so an embedder can
    call it on EVERY load. Directory datasets only, like the pathkey itself.
    """
    base = Path(path).expanduser().absolute()
    if not base.is_dir():
        return False
    keyed = read_pathkey(base)
    here = str(base)
    if keyed:
        return keyed != here
    # No marker (caches predate it): look for hash-tagged artifacts whose tags
    # match neither this spelling's dataset tag nor any of its source-file tags.
    cache_dir = base / '.dataset_explorer_cache'
    if not cache_dir.is_dir():
        return False
    expected = {hashlib.md5(here.encode()).hexdigest()[:8]}
    for p in base.iterdir():
        if p.is_file():
            expected.add(hashlib.md5(str(p.absolute()).encode()).hexdigest()[:8])
    pat = re.compile(r'^.+_([0-9a-f]{8})\.(sets\.gz|metaindex\.parquet|meta\.gz|'
                     r'neardupe\.gz|neardupe-sig\.npy|neardupe-card\.npy)$')
    tags = {m.group(1) for p in cache_dir.iterdir() if (m := pat.match(p.name))}
    return bool(tags - expected)


def migrate_cache(path: str) -> Dict[str, Any]:
    """Adopt caches, temps, sets, and near-dupe artifacts after a dataset MOVED.

    Every cache filename embeds an 8-hex md5 of the keyed file's ABSOLUTE path, so
    copying a dataset (with its tmp/ and .dataset_explorer_cache/) to a new location
    orphans everything: same content, unrecognized names, and the explorer silently
    re-derives days of work. This renames old-hash artifacts to the new location's
    hashes. It is safe by construction: content is still re-validated by the normal
    loaders (metadata caches by size+mtime, result sets by record count, near-dupe
    artifacts by parameters), so a wrong adoption is re-derived, never trusted.
    """
    # absolute(), NEVER resolve(): the explorer keys every cache on the path AS GIVEN
    # (str(p.absolute())). resolve() follows symlinks and Windows mapped drives to a
    # different spelling (W:\... -> \\nas\share\...), which silently renames artifacts
    # to hashes nothing will ever look up. Migration must hash exactly the string the
    # explorer will hash when the user opens this same path.
    base = Path(path).expanduser().absolute()
    if not base.exists():
        raise FileNotFoundError(f"Not found: {base}")

    report: Dict[str, List] = {'renamed': [], 'duplicates_removable': [],
                               'ambiguous': [], 'mtime_fixed': [], 'notes': []}

    def adopt(directory: Path, stem: str, suffix: str, new_tag: str,
              loose_stem: bool = False,
              exclude_stems: Optional[set] = None) -> Optional[Path]:
        """Rename <stem>_<8hex><suffix> -> <stem>_<new_tag><suffix> in directory.

        loose_stem also matches artifacts whose stem differs (the dataset DIRECTORY
        was renamed by the move, so old artifacts carry the old directory name).
        exclude_stems guards loose matching: a SINGLE-FILE dataset that lives
        inside this directory keeps its artifacts in the same cache dir under its
        own file stem -- those belong to a sibling dataset, not to an old
        generation of this one, and must never be swept into a rename or a
        conflict. (Learned the embarrassing way: a 100k-slice test file's caches
        showed up as 'Keep B' options against the real 13M-doc artifacts.)
        """
        target = directory / f"{stem}_{new_tag}{suffix}"
        if not directory.is_dir():
            return target if target.exists() else None
        stem_pat = r'(?P<stem>.+)' if loose_stem else f"(?P<stem>{re.escape(stem)})"
        pat = re.compile(rf"^{stem_pat}_([0-9a-f]{{8}}){re.escape(suffix)}$")
        cands = [p for p in directory.iterdir()
                 if (m := pat.match(p.name)) and p.name != target.name
                 and m.group('stem') not in (exclude_stems or ())]
        if target.exists():
            # A fresh artifact was already re-derived at the new location (e.g. the
            # user let a reload run before migrating). Keep it; the old-hash copies
            # are now dead weight the user may want to delete.
            for c in cands:
                report['duplicates_removable'].append(
                    f"{c} ({c.stat().st_size / 1e6:.0f} MB)")
            return target
        if len(cands) == 1:
            cands[0].rename(target)
            report['renamed'].append(f"{cands[0].name} -> {target.name}")
            print(f"  adopted: {cands[0].name} -> {target.name}")
            return target
        if len(cands) > 1:
            # Structured, so an embedder can present a chooser instead of a chore:
            # the tool cannot know which generation is current, but a human with
            # dates and sizes can, in seconds.
            report['ambiguous'].append({
                'directory': str(directory), 'stem': stem, 'suffix': suffix,
                'target': str(target),
                'candidates': [{'path': str(c), 'name': c.name,
                                'mtime': c.stat().st_mtime,
                                'size_mb': c.stat().st_size / 1e6}
                               for c in sorted(cands, key=lambda p: -p.stat().st_mtime)],
            })
            print(f"  CONFLICT: {len(cands)} candidates for {target.name} -- "
                  f"needs a keep/discard decision")
        return None

    # ---- dataset-level artifacts (sets, metaindex, near-dupe) ----
    ds_cache = (base if base.is_dir() else base.parent) / '.dataset_explorer_cache'
    ds_stem = base.name if base.is_dir() else base.stem
    ds_tag = hashlib.md5(str(base).encode()).hexdigest()[:8]
    sibling_stems: set = set()
    if base.is_dir():
        for p in base.iterdir():
            if p.is_file():
                sibling_stems.add(p.stem)      # 'x.jsonl' for x.jsonl.zst; 'x' for x.jsonl
        sibling_stems.discard(ds_stem)
    for suffix in ('.sets.gz', '.metaindex.parquet', '.neardupe.gz',
                   '.neardupe-sig.npy', '.neardupe-card.npy'):
        adopt(ds_cache, ds_stem, suffix, ds_tag, loose_stem=base.is_dir(),
              exclude_stems=sibling_stems)

    # ---- per-source-file artifacts (metadata caches, decompressed temps) ----
    if base.is_dir():
        sources = []
        has_npy = False
        for p in sorted(base.rglob('*')):
            if not p.is_file():
                continue
            rel_dirs = p.relative_to(base).parts[:-1]
            if any(d in ('tmp', '.dataset_explorer_cache') or d.startswith('.')
                   for d in rel_dirs):
                continue
            if p.suffix.lower() == '.npy':
                has_npy = True
            elif (p.suffix.lower() in ('.parquet', '.jsonl', '.json')
                    or p.name.lower().endswith('.jsonl.zst')):
                sources.append(p)
        if has_npy:
            report['notes'].append(
                ".npy decode temps are keyed on tokenizer settings too and were not "
                "migrated; they will be re-derived on open (or reopen with raw shards).")
    else:
        sources = [base]

    # First gather everything into the consolidated layout (legacy scattered
    # caches from before the layout change, including in subdirectories), THEN
    # adopt path-hash names within it.
    n_consolidated = consolidate_cache_layout(base, sources)
    if n_consolidated:
        report['notes'].append(f"consolidated {n_consolidated} legacy-layout file(s)")

    n_src = len(sources)
    for i, src in enumerate(sources):
        report_progress('migrate caches', i, n_src, main=True)
        src_tag = hashlib.md5(str(src.absolute()).encode()).hexdigest()[:8]

        # metadata cache: <stem minus .jsonl>_<tag>.meta.gz, keyed on the SOURCE file
        meta_stem = src.stem
        if meta_stem.endswith('.jsonl'):
            meta_stem = meta_stem[:-6]
        meta = adopt(ds_cache, meta_stem, '.meta.gz', src_tag)

        # decompressed/converted temp in the consolidated tmp/: <stem>_<tag><ext>
        temp = None
        name_lc = src.name.lower()
        if name_lc.endswith('.jsonl.zst'):
            temp = adopt(ds_cache / 'tmp', src.stem[:-6], '.jsonl', src_tag)
        elif src.suffix.lower() == '.json':
            temp = adopt(ds_cache / 'tmp', src.stem, '.jsonl', src_tag)

        # A copy can scramble the mtime ORDER the caches validate against. The user
        # invoked migration precisely because these files belong together, so restore
        # the invariants: temp newer than source; cached mtime equal to source's.
        try:
            if temp is not None and temp.exists() \
                    and temp.stat().st_mtime <= src.stat().st_mtime:
                os.utime(temp, None)
                report['mtime_fixed'].append(temp.name)
            if meta is not None and meta.exists():
                with gzip.open(meta, 'rb') as f:
                    blob = pickle.load(f)
                st = src.stat()
                if blob.get('file_size') == st.st_size \
                        and blob.get('file_mtime') != st.st_mtime:
                    # This rewrite re-gzips the line-position index, which is why
                    # migration takes real time on a big corpus -- announce it.
                    print(f"  rewriting cache timestamp: {meta.name} "
                          f"({meta.stat().st_size / 1e6:.1f} MB)...")
                    blob['file_mtime'] = st.st_mtime
                    with gzip.open(meta, 'wb', compresslevel=6) as f:
                        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
                    report['mtime_fixed'].append(meta.name)
        except Exception as e:
            report['notes'].append(f"mtime fixup failed for {src.name}: {e}")
            print(f"  NOTE: mtime fixup failed for {src.name}: {e}")
    report_progress('migrate caches', n_src, n_src, main=True)

    for item in report['duplicates_removable']:
        print(f"  duplicate (safe to delete): {item}")
    for c in report['ambiguous']:
        print(f"  CONFLICT {c['target']}:")
        for cand in c['candidates']:
            print(f"    {cand['name']}  modified "
                  f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(cand['mtime']))}"
                  f"  {cand['size_mb']:.1f} MB")
    for item in report['notes']:
        print(f"  NOTE: {item}")
    print(f"Cache migration: {len(report['renamed'])} artifact(s) adopted, "
          f"{len(report['mtime_fixed'])} timestamp(s) repaired for {base}")
    # Only mark the tree as keyed-for-this-path once NOTHING is left ambiguous:
    # a premature marker would make the next load skip adoption and silently
    # open without the conflicted artifacts (e.g. missing result sets).
    if not report['ambiguous']:
        write_pathkey(base)
    return report


class TemporaryFileManager:
    """Manages temporary files created during decompression."""
    
    def __init__(self):
        self.temp_files = []
        self.temp_dirs = []
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def register_file(self, filepath: Path):
        """Register a temporary file for cleanup."""
        self.temp_files.append(filepath)
    
    def register_dir(self, dirpath: Path):
        """Register a temporary directory for cleanup."""
        self.temp_dirs.append(dirpath)
    
    def cleanup(self):
        """Clean up all registered temporary files and directories."""
        for filepath in self.temp_files:
            try:
                if filepath.exists():
                    filepath.unlink()
                    print(f"Cleaned up temporary file: {filepath.name}")
            except Exception as e:
                print(f"Warning: Could not delete temporary file {filepath}: {e}")
        
        for dirpath in self.temp_dirs:
            try:
                # rmdir, NOT rmtree: the tmp/ directory is shared by every decode and
                # decompression beside a given source, so a recursive delete would take
                # out other datasets' cached work. Removing it only when empty means we
                # clean up after ourselves and nobody else.
                if dirpath.exists() and not any(dirpath.iterdir()):
                    dirpath.rmdir()
                    print(f"Removed empty temporary directory: {dirpath}")
            except Exception as e:
                print(f"Warning: Could not remove temporary directory {dirpath}: {e}")


# Global temporary file manager
temp_manager = TemporaryFileManager()


def decompress_zst_file(zst_filepath: Path, console: Optional[Console] = None,
                        clear_temp: bool = False,
                        temp_dir: Optional[Path] = None) -> Path:
    """
    Decompress a .zst file to a temporary location.
    Returns the path to the decompressed file.

    The result PERSISTS by default. It is a proper cache -- hash-keyed on the source path
    and invalidated by mtime -- so deleting it on exit would throw away expensive work and
    guarantee the reuse check below never fires across runs. Pass clear_temp=True to
    delete it at exit instead.
    """
    if not ZSTD_AVAILABLE:
        raise ImportError(
            "zstandard library not installed. Please install it to work with .zst files:\n"
            "pip install zstandard"
        )
    
    # Get file size for progress tracking
    file_size = zst_filepath.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    # Default (legacy) location beside the source; the explorer passes the
    # dataset's consolidated cache tmp/ instead.
    if temp_dir is None:
        temp_dir = zst_filepath.parent / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_manager.register_dir(temp_dir)
    
    # Use hash to avoid conflicts with multiple files
    file_hash = hashlib.md5(str(zst_filepath.absolute()).encode()).hexdigest()[:8]

    # Extension goes LAST: "<name>_<hash>.jsonl", matching convert_npy_to_jsonl and
    # convert_json_array_to_jsonl. The previous form appended the hash after the
    # extension ("ao3_1000001-1100000.jsonl_741c799c"), leaving the file with no usable
    # suffix -- invisible to _list_directory_files, unopenable by any extension-driven
    # tool, and undetectable by _detect_file_type if it were ever reached.
    stem = zst_filepath.stem                      # "foo.jsonl" for "foo.jsonl.zst"
    ext = ''
    for candidate in ('.jsonl', '.json'):
        if stem.lower().endswith(candidate):
            stem, ext = stem[:-len(candidate)], candidate
            break
    temp_filepath = temp_dir / f"{stem}_{file_hash}{ext}"

    # Migrate a file written under the old scheme rather than decompressing it again.
    # The hash is derived from the source path and is unchanged, so this is an exact
    # match, and a rename within the same tmp/ directory is atomic and free -- worth
    # doing because these now persist between runs and can be very large.
    legacy_path = temp_dir / f"{zst_filepath.stem}_{file_hash}"
    if ext and legacy_path.exists() and not temp_filepath.exists():
        legacy_path.rename(temp_filepath)
        msg = f"Renamed legacy temp file {legacy_path.name} -> {temp_filepath.name}"
        if console and RICH_AVAILABLE:
            console.print(f"[green]{msg}[/green]")
        else:
            print(msg)


    # Check if already decompressed
    if temp_filepath.exists():
        # Verify it's still valid
        original_mtime = zst_filepath.stat().st_mtime
        temp_mtime = temp_filepath.stat().st_mtime

        if temp_mtime > original_mtime:
            if _decompressed_temp_complete(temp_filepath):
                if console and RICH_AVAILABLE:
                    console.print(f"[green]Using existing decompressed file: {temp_filepath.name}[/green]")
                else:
                    print(f"Using existing decompressed file: {temp_filepath.name}")
                return temp_filepath
            # Pre-atomic-write versions could leave a truncated temp under the
            # final name if the job was killed mid-decompress. Heal it here.
            print(f"Existing decompressed file {temp_filepath.name} looks INCOMPLETE "
                  f"(truncated by a killed job?); re-decompressing.")
            temp_filepath.unlink()
        else:
            # Original file is newer, re-decompress
            temp_filepath.unlink()
    
    print(f"Decompressing {file_size_mb:.1f} MB .zst file...")
    print(f"Temporary location: {temp_filepath}")

    if clear_temp:
        temp_manager.register_file(temp_filepath)

    # Write to a .part file and rename ONLY on completion. A decompression killed
    # mid-write (server restart, aborted job) used to leave a plausible-looking
    # partial temp that the reuse check would trust -- silently truncating the
    # dataset and, downstream, invalidating every stored result set against a
    # wrong record count. The final name now always means "complete".
    part_path = temp_filepath.with_name(temp_filepath.name + '.part')
    
    # Decompress with progress tracking
    dctx = zstd.ZstdDecompressor()
    
    if RICH_AVAILABLE and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[progress.filesize]{task.fields[size_info]}"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                "Decompressing...", 
                total=file_size,
                size_info=f"0.0 / {file_size_mb:.1f} MB"
            )
            
            bytes_read = 0
            bytes_written = 0
            
            with open(zst_filepath, 'rb') as infile:
                with open(part_path, 'wb') as outfile:
                    # Use streaming decompression
                    reader = dctx.stream_reader(infile)
                    
                    while True:
                        chunk = reader.read(1024 * 1024)  # Read 1MB chunks
                        if not chunk:
                            break
                        
                        outfile.write(chunk)
                        bytes_written += len(chunk)
                        
                        # Update based on input file position
                        bytes_read = infile.tell()
                        progress.update(
                            task, 
                            completed=bytes_read,
                            size_info=f"{bytes_read/(1024*1024):.1f} / {file_size_mb:.1f} MB"
                        )
            
            progress.update(task, completed=file_size)
            decompressed_size_mb = bytes_written / (1024 * 1024)
            print(f"Decompressed size: {decompressed_size_mb:.1f} MB (ratio: {decompressed_size_mb/file_size_mb:.1f}x)")
    
    elif TQDM_AVAILABLE:
        bytes_written = 0
        
        with open(zst_filepath, 'rb') as infile:
            with open(temp_filepath, 'wb') as outfile:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc="Decompressing") as pbar:
                    dctx = zstd.ZstdDecompressor()
                    reader = dctx.stream_reader(infile)
                    
                    last_position = 0
                    while True:
                        chunk = reader.read(1024 * 1024)  # Read 1MB chunks
                        if not chunk:
                            break
                        
                        outfile.write(chunk)
                        bytes_written += len(chunk)
                        
                        # Update progress based on input file position
                        current_position = infile.tell()
                        pbar.update(current_position - last_position)
                        last_position = current_position
        
        decompressed_size_mb = bytes_written / (1024 * 1024)
        print(f"Decompressed size: {decompressed_size_mb:.1f} MB (ratio: {decompressed_size_mb/file_size_mb:.1f}x)")
    
    else:
        # No progress bar libraries available
        print("Decompressing... (this may take a while for large files)")
        
        bytes_written = 0
        with open(zst_filepath, 'rb') as infile:
            with open(part_path, 'wb') as outfile:
                dctx = zstd.ZstdDecompressor()
                reader = dctx.stream_reader(infile)

                chunk_count = 0
                while True:
                    chunk = reader.read(1024 * 1024)  # Read 1MB chunks
                    if not chunk:
                        break
                    
                    outfile.write(chunk)
                    bytes_written += len(chunk)
                    chunk_count += 1
                    report_progress('decompress', infile.tell(), file_size,
                                    unit='bytes')

                    if chunk_count % 100 == 0:  # Update every 100MB
                        print(f"  Processed {bytes_written/(1024*1024):.1f} MB...")
        
        decompressed_size_mb = bytes_written / (1024 * 1024)
        print(f"Decompressed size: {decompressed_size_mb:.1f} MB (ratio: {decompressed_size_mb/file_size_mb:.1f}x)")
    
    part_path.replace(temp_filepath)
    print(f"Decompression complete: {temp_filepath.name}")
    return temp_filepath


def _decompressed_temp_complete(temp_filepath: Path) -> bool:
    """Cheap completeness check for a decompressed JSONL temp.

    A decompression killed mid-write truncates the stream mid-line (writes go
    out in 1 MB chunks, and lines are KBs), so a complete file ends with a
    newline and its final full line parses as JSON. One seek + one small read;
    a false negative merely re-decompresses, a false positive is ~impossible
    for a chunk-boundary truncation to fake.
    """
    try:
        size = temp_filepath.stat().st_size
        if size == 0:
            return False
        with open(temp_filepath, 'rb') as f:
            f.seek(max(0, size - (1 << 20)))
            tail = f.read()
        if not tail.endswith(b'\n'):
            return False
        lines = tail.split(b'\n')
        if len(lines) >= 3:              # a full last line lies inside the window
            json.loads(lines[-2])
        return True
    except Exception:
        return False


def _line_index_matches(path: Path, line_positions, num_rows) -> bool:
    """True if a cached line index provably describes THIS file.

    The cache is keyed to the SOURCE file's mtime/size, but the index was built
    over the decompressed temp -- which can have been replaced (e.g. after a
    truncated temp was healed) without the source changing. Proof: the index's
    last entry must be the offset of the file's final line, i.e. reading that
    line lands exactly at EOF.
    """
    try:
        if not line_positions:
            return True
        if num_rows is not None and len(line_positions) != num_rows:
            return False
        size = path.stat().st_size
        last = int(line_positions[-1])
        if last >= size:
            return False
        with open(path, 'rb') as f:
            f.seek(last)
            f.readline()
            return f.tell() == size
    except Exception:
        return False


def _peek_first_nonws_char(filepath: Path, max_bytes: int = 4096) -> str:
    """Return the first non-whitespace character of a text file, or '' if empty."""
    with open(filepath, 'rb') as f:
        chunk = f.read(max_bytes)
    try:
        text = chunk.decode('utf-8', errors='replace')
    except Exception:
        return ''
    for ch in text:
        if not ch.isspace():
            return ch
    return ''


def convert_json_array_to_jsonl(json_filepath: Path, console: Optional[Console] = None,
                                clear_temp: bool = False,
                                temp_dir: Optional[Path] = None) -> Path:
    """Convert a top-level JSON array file to a JSONL temp file. Returns its path.

    Loads the array fully via json.load (memory cost ≈ 2-3x file size while parsing).
    Caches the converted file in <source_dir>/tmp/ keyed on the original path; reuses
    if newer than the source. Persists by default; clear_temp=True deletes it at exit.
    """
    file_size = json_filepath.stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    if temp_dir is None:
        temp_dir = json_filepath.parent / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_manager.register_dir(temp_dir)

    file_hash = hashlib.md5(str(json_filepath.absolute()).encode()).hexdigest()[:8]
    temp_filepath = temp_dir / f"{json_filepath.stem}_{file_hash}.jsonl"

    if temp_filepath.exists():
        if temp_filepath.stat().st_mtime > json_filepath.stat().st_mtime:
            msg = f"Using existing JSONL conversion: {temp_filepath.name}"
            if console and RICH_AVAILABLE:
                console.print(f"[green]{msg}[/green]")
            else:
                print(msg)
            return temp_filepath
        temp_filepath.unlink()

    print(f"Converting JSON array ({file_size_mb:.1f} MB) to JSONL...")
    print(f"Temporary location: {temp_filepath}")
    if clear_temp:
        temp_manager.register_file(temp_filepath)

    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array at the top level of {json_filepath.name}, "
            f"got {type(data).__name__}. Use .jsonl format for line-delimited records."
        )

    n = len(data)
    print(f"Writing {n:,} records as JSONL...")
    # .part + rename: same crash-safety as decompress_zst_file -- the final
    # name must always mean "complete", or a killed job poisons the cache.
    part_path = temp_filepath.with_name(temp_filepath.name + '.part')
    with open(part_path, 'w', encoding='utf-8') as f:
        for i, record in enumerate(data):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if i % 10_000 == 0:
                report_progress('convert json array', i, n)
    part_path.replace(temp_filepath)

    print(f"Conversion complete: {temp_filepath.name}")
    return temp_filepath


class MetadataCache:
    """Handles caching and retrieval of dataset metadata."""
    
    def __init__(self, data_filepath: Path, original_filepath: Optional[Path] = None,
                 cache_dir: Optional[Path] = None):
        """
        Initialize cache for a data file.

        Args:
            data_filepath: Path to the actual data file (may be decompressed temp file)
            original_filepath: Path to the original file if data_filepath is a temp file
            cache_dir: Where cache files live; defaults to the legacy per-source-dir
                location (the explorer passes the dataset's consolidated cache root)
        """
        self.data_filepath = data_filepath
        self.original_filepath = original_filepath

        # Use original filepath for cache if this is a decompressed file
        cache_filepath = original_filepath if original_filepath else data_filepath
        self.cache_dir = cache_dir if cache_dir is not None \
            else cache_filepath.parent / '.dataset_explorer_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a unique cache filename based on the original file
        file_hash = hashlib.md5(str(cache_filepath.absolute()).encode()).hexdigest()[:8]
        cache_stem = cache_filepath.stem
        # Remove .jsonl from stem if present (for .jsonl.zst files)
        if cache_stem.endswith('.jsonl'):
            cache_stem = cache_stem[:-6]
        self.cache_filepath = self.cache_dir / f"{cache_stem}_{file_hash}.meta.gz"
    
    def is_valid(self) -> bool:
        """Check if cached metadata exists and is still valid."""
        if not self.cache_filepath.exists():
            return False
        
        try:
            # Load cache header to check validity
            with gzip.open(self.cache_filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check against original file if this is a decompressed file
            check_filepath = self.original_filepath if self.original_filepath else self.data_filepath
            current_stats = check_filepath.stat()
            cached_mtime = cache_data.get('file_mtime')
            cached_size = cache_data.get('file_size')
            
            if cached_mtime != current_stats.st_mtime or cached_size != current_stats.st_size:
                print(f"Cache invalidated: file has been modified")
                return False
            
            return True
        
        except Exception as e:
            print(f"Cache validation failed: {e}")
            return False
    
    def load(self) -> Optional[Dict[str, Any]]:
        """Load cached metadata if valid."""
        if not self.is_valid():
            return None
        
        try:
            with gzip.open(self.cache_filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            print(f"Loaded metadata from cache: {self.cache_filepath.name}")
            return cache_data
        
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return None
    
    def save(self, metadata: Dict[str, Any], line_positions: Optional[List[int]] = None):
        """Save metadata to cache."""
        try:
            # Use original file stats if this is a decompressed file
            check_filepath = self.original_filepath if self.original_filepath else self.data_filepath
            file_stats = check_filepath.stat()
            
            cache_data = {
                'file_mtime': file_stats.st_mtime,
                'file_size': file_stats.st_size,
                'file_path': str(check_filepath.absolute()),
                'cache_version': '1.1',  # Updated version for zst support
                'cached_at': time.time(),
                'metadata': metadata,
                'line_positions': line_positions  # For JSONL files
            }
            
            # Use gzip compression for potentially large line position arrays
            with gzip.open(self.cache_filepath, 'wb', compresslevel=6) as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            cache_size_mb = self.cache_filepath.stat().st_size / (1024 * 1024)
            print(f"Saved metadata to cache: {self.cache_filepath.name} ({cache_size_mb:.2f} MB)")
            
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def clear(self):
        """Clear the cache for this file."""
        if self.cache_filepath.exists():
            self.cache_filepath.unlink()
            print(f"Cleared cache: {self.cache_filepath.name}")


def load_npy_tokenizer(kind: str, path: str, special_tokens: Optional[str] = None):
    """Load a tokenizer via mara's tokenizer_abstraction (same one train_mara.py uses),
    so BOS and decode are IDENTICAL to how the .npy was produced. common_fsdp2 is added
    to sys.path relative to this file (…/code/tools -> …/code/common_fsdp2)."""
    import sys as _sys, os as _os
    cf = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'common_fsdp2')
    if _os.path.isdir(cf) and cf not in _sys.path:
        _sys.path.insert(0, cf)
    try:
        from tokenizer_abstraction import get_tokenizer  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Reading .npy token shards needs mara's tokenizer_abstraction (common_fsdp2). "
            f"Could not import it ({e}). Run from the code/ tree or add common_fsdp2 to PYTHONPATH."
        )
    return get_tokenizer(kind, path=_os.path.expanduser(path),
                         special_tokens=_os.path.expanduser(special_tokens) if special_tokens else None)


def convert_npy_to_jsonl(npy_filepath: Path, tokenizer, console: Optional[Console] = None,
                         max_docs: Optional[int] = None, clear_temp: bool = False,
                         temp_dir: Optional[Path] = None) -> Path:
    """Decode a tokenized .npy shard into a temp .jsonl of {"text": <decoded doc>} records,
    so the existing jsonl machinery handles it. Docs are split on the tokenizer's BOS id
    (NOT hardcoded); dtype comes from the .npy header (NOT hardcoded). Hash-cached in a
    sibling tmp/ dir, mirroring decompress_zst_file. --max-docs caps the decode for a quick peek.

    The decode PERSISTS by default so subsequent opens are instant cache hits: decoding a
    large shard set is the single most expensive thing this tool does, and the cached copy
    is validated against the source mtime. clear_temp=True restores exit-cleanup."""
    bos = tokenizer.bos_id
    arr = np.load(npy_filepath, mmap_mode='r')            # dtype straight from the file header
    if arr.ndim != 1:
        raise ValueError(f"{npy_filepath.name}: expected a 1-D token array, got shape {arr.shape}")

    if temp_dir is None:
        temp_dir = npy_filepath.parent / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_manager.register_dir(temp_dir)
    tag = hashlib.md5((str(npy_filepath.absolute()) + f"|bos{bos}|max{max_docs}").encode()).hexdigest()[:8]
    out_path = temp_dir / f"{npy_filepath.stem}_{tag}.jsonl"
    if out_path.exists() and out_path.stat().st_mtime > npy_filepath.stat().st_mtime:
        msg = f"Using cached decode: {out_path.name}" + ("" if clear_temp else " (persisted)")
        (console.print(f"[green]{msg}[/green]") if (console and RICH_AVAILABLE) else print(msg))
        return out_path

    bos_pos = np.where(arr == bos)[0]
    # doc starts; if the stream doesn't begin with BOS, treat position 0 as the first doc start
    starts = bos_pos.tolist()
    if not starts or starts[0] != 0:
        starts = [0] + starts
    bounds = starts + [len(arr)]
    n_docs = len(starts) if max_docs is None else min(len(starts), max_docs)
    msg = f"Decoding {npy_filepath.name}: {n_docs:,} docs (bos={bos}, dtype={arr.dtype}) -> {out_path.name}"
    (console.print(f"[cyan]{msg}[/cyan]") if (console and RICH_AVAILABLE) else print(msg))

    tmp_write = out_path.with_suffix('.jsonl.part')
    # This decode used to run SILENT in every display mode -- on a large shard that
    # reads as a hang. Milestone prints every ~10% serve the CLI; report_progress
    # serves embedders at a finer cadence.
    milestone = max(1, n_docs // 10)
    with open(tmp_write, 'w') as f:
        for i in range(n_docs):
            s, e = bounds[i], bounds[i + 1]
            ids = arr[s + 1:e].tolist() if (s < len(arr) and int(arr[s]) == bos) else arr[s:e].tolist()
            text = tokenizer.decode(ids)
            f.write(json.dumps({'text': text}, ensure_ascii=True) + '\n')
            if i % 200 == 0:
                report_progress(f'decode {npy_filepath.name}', i, n_docs)
            if i and i % milestone == 0:
                print(f"  decoded {i:,}/{n_docs:,} docs ({i / n_docs * 100:.0f}%)...")
    tmp_write.replace(out_path)
    if clear_temp:
        temp_manager.register_file(out_path)
    return out_path


def npy_doc_spans(npy_filepath: Path, bos: int) -> Tuple[np.ndarray, np.ndarray]:
    """(starts, ends) token offsets per document, INCLUDING each document's BOS.

    These are the spans to copy when REWRITING a shard: pre_tokenize.py's doc-aligned
    writer guarantees every shard begins with BOS and holds only whole documents, and
    copying BOS-inclusive spans preserves both properties in the output.

    Splits on the tokenizer's BOS id, exactly like convert_npy_to_jsonl, but WITHOUT
    decoding. Uses mmap so only the token stream is touched, never materialized.

    Caveat: shards written with `pre_tokenize.py --legacy-river` split documents ACROSS
    shard boundaries, so a document's tail begins a shard with no BOS and is counted here
    as a separate document. Legacy trees are identifiable by the absence of
    manifest_{split}.json.
    """
    arr = np.load(npy_filepath, mmap_mode='r')
    if arr.ndim != 1:
        raise ValueError(f"{npy_filepath.name}: expected a 1-D token array, got shape {arr.shape}")
    starts = np.flatnonzero(np.asarray(arr) == bos)
    if starts.size == 0 or starts[0] != 0:
        starts = np.concatenate(([0], starts))
    ends = np.concatenate((starts[1:], [len(arr)]))
    return starts.astype(np.int64), ends.astype(np.int64)


def npy_doc_bounds(npy_filepath: Path, bos: int) -> Tuple[np.ndarray, np.ndarray]:
    """(starts, ends) with each document's BOS stripped -- content spans for shingling.

    Same splitting as npy_doc_spans, so document COUNT and ORDER are identical; only the
    start offset differs. Keeping one splitter means the prune and the detector can never
    disagree about which document is record N.
    """
    starts, ends = npy_doc_spans(npy_filepath, bos)
    arr = np.load(npy_filepath, mmap_mode='r')
    doc_starts = np.where(np.asarray(arr)[starts] == bos, starts + 1, starts)
    return doc_starts.astype(np.int64), ends.astype(np.int64)


def iter_npy_token_docs(npy_filepath: Path, bos: int, max_docs: Optional[int] = None):
    """Yield (local_doc_index, token_ids) straight from a tokenized .npy shard.

    The explorer's normal .npy path decodes every document back to text via
    convert_npy_to_jsonl so the JSONL machinery can address it. For deduplication that
    decode is not just wasted work, it is LOSSY: detokenize -> re-tokenize does not
    round-trip, so shingles computed on decoded text describe a document the model never
    saw. Shingling token IDs directly avoids both problems and reads at disk speed.

    The tokenizer is needed ONLY for `bos` (document boundaries) -- nothing is decoded.
    """
    arr = np.load(npy_filepath, mmap_mode='r')
    starts, ends = npy_doc_bounds(npy_filepath, bos)
    n = len(starts) if max_docs is None else min(len(starts), max_docs)
    for i in range(n):
        yield i, np.asarray(arr[starts[i]:ends[i]])


class DatasetExplorer:
    """Main class for exploring large dataset files."""
    
    def __init__(self, filepath: str, max_display_width: int = 200, quick_mode: bool = False,
                 no_cache: bool = False, rebuild_cache: bool = False,
                 tok_kind: Optional[str] = None, tok_path: Optional[str] = None,
                 special_tokens: Optional[str] = None, npy_max_docs: Optional[int] = None,
                 clear_temp: bool = False, dedup_only: bool = False,
                 text_field: Optional[str] = None, non_interactive: bool = False,
                 recursive: bool = False):
        self.original_filepath = Path(filepath)
        if not self.original_filepath.exists():
            raise FileNotFoundError(f"File or directory not found: {filepath}")

        # Consolidated cache layout: everything derived from this dataset --
        # metadata caches, sets, indexes, near-dupe artifacts, and (under tmp/)
        # decompressions/conversions/decodes -- lives in ONE directory.
        self.cache_root = dataset_cache_root(self.original_filepath)
        self.temp_root = self.cache_root / 'tmp'

        self.max_display_width = max_display_width
        self.quick_mode = quick_mode
        self.no_cache = no_cache
        self.rebuild_cache = rebuild_cache
        # .npy token-shard support: tokenizer (for BOS + decode) is required only for .npy sources
        self.tok_kind = tok_kind
        self.tok_path = tok_path
        self.special_tokens = special_tokens
        self.npy_max_docs = npy_max_docs
        self.clear_temp = clear_temp
        # dedup_only: read .npy token shards RAW instead of decoding them to JSONL.
        # Near-duplicate detection shingles token ids directly, so the decode is both
        # wasted work and the single most expensive step of opening a large shard set.
        # Record-level commands (record/sample/findall) are unavailable in this mode.
        self.dedup_only = dedup_only
        self._npy_tokenizer = None
        # Bounded cache of per-shard document spans, so on-demand decoding of a single
        # record does not rescan the shard for BOS positions on every lookup.
        self._npy_bounds_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        # Explicit body-field designation (--field). The metadata indexer excludes it and
        # counts words from it; without it the body is detected by value length.
        self.text_field = text_field
        # non_interactive: never call input(). Embedders (the web server) have no stdin;
        # the large-JSONL option menu auto-selects "build complete index" instead, because
        # random access is the one capability every embedding needs.
        self.non_interactive = non_interactive
        # recursive: walk subdirectories when the dataset is a directory. OPT-IN:
        # turning it on changes file discovery and therefore global record
        # numbering, which invalidates saved sets -- existing datasets must
        # never change shape because the tool got smarter.
        self.recursive = recursive
        self._metaindex = None
        self.result_sets: Dict[str, Dict[str, Any]] = {}
        self._word_id_memo: Dict[str, int] = {}
        self.dupe_state: Optional[Dict[str, Any]] = None
        self.console = Console() if RICH_AVAILABLE else None

        # Common state
        self.data = None
        self.metadata: Dict[str, Any] = {}
        self.line_positions: Optional[List[int]] = None  # Single-file JSONL only
        self.search_state: Optional[Dict[str, Any]] = None
        self.full_display = False
        self.is_compressed = False
        self.is_json_array = False
        self.is_directory = self.original_filepath.is_dir()

        # Multi-file state (populated only when is_directory=True)
        self.source_files: List[Path] = []
        self.working_files: List[Path] = []
        self.file_record_counts: List[int] = []
        self.file_line_positions: List[Optional[List[int]]] = []
        self.file_metadata_list: List[Dict[str, Any]] = []
        self.file_caches: List[Optional['MetadataCache']] = []
        self.cum_record_counts: List[int] = [0]

        if self.is_directory:
            keyed_for = read_pathkey(self.original_filepath)
            here = str(self.original_filepath.absolute())
            if keyed_for and keyed_for != here:
                # Caches exist but are keyed for a different spelling of this
                # directory (moved dataset, or mapped-drive vs UNC). Everything
                # below this point would silently re-derive days of work.
                print(f"WARNING: caches here are keyed for {keyed_for!r} but the "
                      f"dataset is being opened as {here!r}. Run migrate_cache "
                      f"(--migrate-cache / 'adopt moved caches') to reuse them.")
            self._init_multi_file()
        else:
            self._init_single_file()

        # Record the spelling these caches are keyed for, so a future move is
        # detected up front instead of discovered via a surprise re-decompression.
        write_pathkey(self.original_filepath)

        # After metadata, so stored sets can be validated against the record count.
        self._load_sets()

    def _get_npy_tokenizer(self):
        """Lazily load the tokenizer for .npy sources; errors clearly if not specified."""
        if self._npy_tokenizer is None:
            # tok_path is optional: tiktoken/claude kinds resolve without one.
            if not self.tok_kind:
                raise ValueError(
                    "Reading .npy token shards requires a tokenizer. Pass "
                    "--tok-kind (e.g. llama, tiktoken) and, if the kind needs it, "
                    "--tok-path <tokenizer dir/model> "
                    "(plus optional --special-tokens <tokenizer_config.json>).")
            self._npy_tokenizer = load_npy_tokenizer(
                self.tok_kind, self.tok_path or '', self.special_tokens)
        return self._npy_tokenizer

    def _init_single_file(self):
        """Set up the explorer for a single source file."""
        self.filepath = self.original_filepath  # May be replaced with a temp file
        consolidate_cache_layout(self.original_filepath, [self.original_filepath])

        if self.original_filepath.suffix.lower() == '.zst':
            self.is_compressed = True
            if '.jsonl.zst' in self.original_filepath.name.lower():
                self.filepath = decompress_zst_file(self.original_filepath, self.console,
                                                    self.clear_temp,
                                                    temp_dir=self.temp_root)
                self.file_type = 'jsonl'
            else:
                raise ValueError(
                    f"Unsupported compressed file type: {self.original_filepath.suffix}\n"
                    f"Currently only .jsonl.zst files are supported."
                )
        elif self.original_filepath.suffix.lower() == '.npy':
            if self.dedup_only:
                self.filepath = self.original_filepath
                self.file_type = 'npy'
            else:
                self.filepath = convert_npy_to_jsonl(self.original_filepath, self._get_npy_tokenizer(),
                                                     self.console, self.npy_max_docs, self.clear_temp,
                                                     temp_dir=self.temp_root)
                self.file_type = 'jsonl'
        else:
            self.file_type = self._detect_file_type()
            if self.file_type == 'jsonl' and _peek_first_nonws_char(self.original_filepath) == '[':
                self.is_json_array = True
                self.filepath = convert_json_array_to_jsonl(self.original_filepath, self.console,
                                                            self.clear_temp,
                                                            temp_dir=self.temp_root)
                self.file_type = 'jsonl'

        cache_orig = self.original_filepath if self.filepath != self.original_filepath else None
        # Raw token-shard mode derives its metadata from a mmap scan and never uses the
        # metadata cache. It must not CONSTRUCT one either: the cache path is keyed on the
        # .npy, so it collides with the entry the normal (decoding) path writes, and
        # --rebuild-cache would delete that entry without rebuilding it.
        if self.file_type == 'npy':
            self.cache = None
        else:
            self.cache = None if self.no_cache else MetadataCache(
                self.filepath, cache_orig, cache_dir=self.cache_root)
            if self.rebuild_cache and self.cache:
                self.cache.clear()

        self._load_metadata()

    def _detect_file_type(self) -> str:
        """Detect file type from extension of the working file."""
        suffix = self.filepath.suffix.lower()
        if suffix == '.parquet':
            return 'parquet'
        elif suffix in ['.jsonl', '.json']:
            return 'jsonl'
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    # ---- Multi-file (directory) support ----

    SUPPORTED_EXTENSIONS = ('.parquet', '.jsonl', '.json', '.zst', '.npy')

    def _list_directory_files(self) -> List[Path]:
        """Supported files in the directory. Top-level by default; recursive=True
        walks subdirectories too (HF-style nested shard layouts), skipping the
        explorer's own working dirs (tmp/, .dataset_explorer_cache/) and hidden
        directories, ordered by relative path so record numbering is stable."""
        def _supported(p: Path) -> bool:
            return (p.suffix.lower() in ('.parquet', '.jsonl', '.json', '.npy')
                    or p.name.lower().endswith('.jsonl.zst'))

        base = self.original_filepath
        if self.recursive:
            files = []
            for p in base.rglob('*'):
                if not p.is_file() or not _supported(p):
                    continue
                rel_dirs = p.relative_to(base).parts[:-1]
                if any(d in ('tmp', '.dataset_explorer_cache') or d.startswith('.')
                       for d in rel_dirs):
                    continue
                files.append(p)
            files.sort(key=lambda p: p.relative_to(base).as_posix())
        else:
            files = sorted(p for p in base.iterdir()
                           if p.is_file() and _supported(p))
        # A tokenized-shard directory usually carries metadata sidecars (e.g. manifest_train.json).
        # If any .npy shards are present, treat it as an npy dataset and use ONLY the shards.
        npys = [p for p in files if p.suffix.lower() == '.npy']
        if npys:
            return npys
        # Bare .json files are supported as single-file datasets (JSON arrays),
        # but inside a MIXED directory they are almost always sidecars shipped
        # with the real shards (progress.json, dataset_info.json, ...) -- and one
        # of those, parsed as a dataset, aborts the whole load with a schema
        # mismatch. Use .json only when the directory has nothing else.
        primary = [p for p in files if p.suffix.lower() != '.json']
        return primary if primary else files

    def _prepare_file(self, source_path: Path) -> Dict[str, Any]:
        """Per-file preparation used by multi-file mode.

        Decompresses/converts if needed, loads schema, and (for JSONL) builds the
        line index. Always uses the cache when available; never prompts. Returns a
        dict capturing everything the explorer needs to address records in the file.
        """
        is_compressed = False
        is_json_array = False
        working_path = source_path

        suffix = source_path.suffix.lower()
        name_lc = source_path.name.lower()

        if suffix == '.zst':
            is_compressed = True
            if '.jsonl.zst' in name_lc:
                working_path = decompress_zst_file(source_path, self.console, self.clear_temp,
                                                   temp_dir=self.temp_root)
                file_type = 'jsonl'
            else:
                raise ValueError(
                    f"Unsupported compressed file: {source_path.name}. "
                    f"Only .jsonl.zst is supported."
                )
        elif suffix == '.parquet':
            file_type = 'parquet'
        elif suffix == '.npy':
            if self.dedup_only:
                working_path = source_path
                file_type = 'npy'
            else:
                working_path = convert_npy_to_jsonl(source_path, self._get_npy_tokenizer(),
                                                    self.console, self.npy_max_docs, self.clear_temp,
                                                    temp_dir=self.temp_root)
                file_type = 'jsonl'
        elif suffix in ('.jsonl', '.json'):
            file_type = 'jsonl'
            if _peek_first_nonws_char(source_path) == '[':
                is_json_array = True
                working_path = convert_json_array_to_jsonl(source_path, self.console,
                                                           self.clear_temp,
                                                           temp_dir=self.temp_root)
        else:
            raise ValueError(f"Unsupported file type: {source_path.name}")

        cache_orig = source_path if working_path != source_path else None
        # See _init_single_file: raw token-shard mode must not touch (or clear) the
        # metadata cache entry that the decoding path owns for the same .npy.
        if file_type == 'npy':
            cache = None
        else:
            cache = None if self.no_cache else MetadataCache(
                working_path, cache_orig, cache_dir=self.cache_root)
            if self.rebuild_cache and cache:
                cache.clear()

        columns: Optional[List[str]] = None
        schema: Optional[Dict[str, str]] = None
        num_rows: Optional[int] = None
        line_positions: Optional[List[int]] = None

        if cache:
            cache_data = cache.load()
            if cache_data:
                md = cache_data['metadata']
                columns = md.get('columns')
                schema = md.get('schema')
                num_rows = md.get('num_rows')
                line_positions = cache_data.get('line_positions')
                if (file_type == 'jsonl' and line_positions
                        and not _line_index_matches(working_path, line_positions, num_rows)):
                    print(f"  cached line index does not match {working_path.name} "
                          f"(temp was replaced?); rebuilding")
                    line_positions, num_rows = None, None

        if file_type == 'npy':
            # Raw token shard: document count comes from BOS positions via a mmap scan.
            # No decode, no line index, and no metadata cache (the scan is already cheap).
            num_rows = self._npy_doc_count(working_path)
            columns = ['tokens']
            schema = {'tokens': 'token ids (raw shard)'}
            line_positions = None   # byte offsets are meaningless for a token shard
            cache = None

        elif file_type == 'parquet':
            if columns is None or num_rows is None:
                pf = pq.ParquetFile(working_path)
                num_rows = pf.metadata.num_rows
                arrow_schema = pf.schema_arrow
                columns = list(arrow_schema.names)
                schema = {arrow_schema.field(i).name: str(arrow_schema.field(i).type)
                          for i in range(len(arrow_schema))}
                if cache:
                    cache.save({
                        'columns': columns,
                        'schema': schema,
                        'num_rows': num_rows,
                        'num_columns': len(columns),
                    })

        elif file_type == 'jsonl':
            if columns is None:
                with open(working_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                if first_line:
                    first_record = json.loads(first_line)
                    if not isinstance(first_record, dict):
                        raise ValueError(
                            f"Records in {source_path.name} are not JSON objects "
                            f"(got {type(first_record).__name__})."
                        )
                    columns = list(first_record.keys())
                    schema = {k: type(v).__name__ for k, v in first_record.items()}

            if line_positions is None or num_rows is None:
                num_rows, line_positions = self._build_line_positions_with_progress(working_path)
                if cache:
                    cache.save({
                        'columns': columns,
                        'schema': schema,
                        'num_rows': num_rows,
                        'num_columns': len(columns or []),
                        'has_index': True,
                    }, line_positions)

        return {
            'source_path': source_path,
            'working_path': working_path,
            'file_type': file_type,
            'is_compressed': is_compressed,
            'is_json_array': is_json_array,
            'num_rows': num_rows,
            'columns': columns,
            'schema': schema,
            'line_positions': line_positions,
            'cache': cache,
            'source_size_mb': source_path.stat().st_size / (1024 * 1024),
            'working_size_mb': working_path.stat().st_size / (1024 * 1024),
        }

    def _init_multi_file(self):
        """Set up the explorer for a directory of files (same format, same schema)."""
        candidates = self._list_directory_files()
        if not candidates:
            raise ValueError(
                f"No supported data files found in {self.original_filepath}.\n"
                f"Looked for .parquet, .jsonl, .json, .jsonl.zst (top-level only)."
            )

        if self.console and RICH_AVAILABLE:
            self.console.print(
                f"[cyan]Directory mode: found {len(candidates)} file(s) in "
                f"{self.original_filepath}[/cyan]"
            )
        else:
            print(f"Directory mode: found {len(candidates)} file(s) in {self.original_filepath}")

        consolidate_cache_layout(self.original_filepath, candidates)

        # FAIL-FAST schema pre-flight: read each file's columns CHEAPLY (first
        # line / parquet footer) before any heavy per-file preparation. The
        # per-file check further down remains as the authoritative backstop
        # (it also covers .zst, which would need decompression to peek), but a
        # mismatch must not be discovered at file 101 after an hour of indexing
        # -- which is exactly how a stray progress.json once failed a load.
        def _peek_columns(p: Path) -> Optional[List[str]]:
            try:
                if p.suffix.lower() == '.parquet':
                    return list(pq.ParquetFile(p).schema_arrow.names)
                if p.name.lower().endswith('.zst') or p.suffix.lower() == '.npy':
                    return None
                with open(p, 'r', encoding='utf-8') as f:
                    line = f.readline()
                if not line.strip():
                    return None
                rec = json.loads(line)
                return list(rec.keys()) if isinstance(rec, dict) else None
            except Exception:
                return None          # unreadable here -> full prep will report it

        if len(candidates) > 50:
            print(f"Schema pre-flight: reading the first line of "
                  f"{len(candidates):,} files (catches mismatches before "
                  f"any indexing)...")
        peeked = []
        for i, p in enumerate(candidates):
            if i % 100 == 0:
                report_progress('schema pre-flight', i, len(candidates),
                                note=p.name)
            peeked.append((p, _peek_columns(p)))
        report_progress('schema pre-flight', len(candidates), len(candidates))
        ref = next(((p, c) for p, c in peeked if c is not None), None)
        if ref is not None:
            for p, cols in peeked:
                if cols is not None and cols != ref[1]:
                    raise ValueError(
                        f"Schema mismatch in {p.name}:\n"
                        f"  Expected columns (from {ref[0].name}): {ref[1]}\n"
                        f"  Got: {cols}\n"
                        f"All files in the directory must share the same fields. "
                        f"(Detected before indexing -- nothing was scanned.)")

        canonical_columns: Optional[List[str]] = None
        canonical_schema: Optional[Dict[str, str]] = None
        canonical_type: Optional[str] = None

        for i, source_path in enumerate(candidates, 1):
            try:
                label = str(source_path.relative_to(self.original_filepath))
            except ValueError:
                label = source_path.name
            if self.console and RICH_AVAILABLE:
                self.console.print(f"\n[bold cyan]\\[{i}/{len(candidates)}][/bold cyan] {label}")
            else:
                print(f"\n[{i}/{len(candidates)}] {label}")
            report_progress('load files', i - 1, len(candidates), main=True)

            info = self._prepare_file(source_path)

            if canonical_type is None:
                canonical_type = info['file_type']
            elif info['file_type'] != canonical_type:
                raise ValueError(
                    f"Mixed file types in directory:\n"
                    f"  {candidates[0].name}: {canonical_type}\n"
                    f"  {source_path.name}: {info['file_type']}\n"
                    f"All files must be the same format."
                )

            if canonical_columns is None:
                canonical_columns = info['columns']
                canonical_schema = info['schema']
            elif info['columns'] != canonical_columns:
                raise ValueError(
                    f"Schema mismatch in {source_path.name}:\n"
                    f"  Expected columns (from {candidates[0].name}): {canonical_columns}\n"
                    f"  Got: {info['columns']}\n"
                    f"All files in the directory must share the same fields."
                )

            self.source_files.append(info['source_path'])
            self.working_files.append(info['working_path'])
            self.file_record_counts.append(info['num_rows'])
            self.file_line_positions.append(info['line_positions'])
            self.file_metadata_list.append(info)
            self.file_caches.append(info['cache'])

        report_progress('load files', len(candidates), len(candidates), main=True)

        # Cumulative offsets: cum[i] = total records BEFORE file i; cum[-1] = grand total
        self.cum_record_counts = [0]
        for c in self.file_record_counts:
            self.cum_record_counts.append(self.cum_record_counts[-1] + c)

        self.file_type = canonical_type

        total_source_mb = sum(m['source_size_mb'] for m in self.file_metadata_list)
        total_working_mb = sum(m['working_size_mb'] for m in self.file_metadata_list)

        self.metadata = {
            'file_path': str(self.original_filepath.absolute()),
            'is_directory': True,
            'num_files': len(self.source_files),
            'num_rows': self.cum_record_counts[-1],
            'columns': canonical_columns,
            'schema': canonical_schema,
            'num_columns': len(canonical_columns) if canonical_columns else 0,
            'file_size': total_source_mb,
            'total_source_mb': total_source_mb,
            'total_working_mb': total_working_mb,
            'has_index': (canonical_type == 'jsonl'
                          and all(lp is not None for lp in self.file_line_positions)),
        }

        # Single-file aliases for backward compat with code that hasn't been updated.
        # In multi-file mode these point at the FIRST file as a sensible default,
        # but methods that operate over all files should consult self.is_directory.
        self.filepath = self.working_files[0]
        self.line_positions = None
        self.cache = None

    def _global_to_local(self, global_idx: int) -> Tuple[int, int]:
        """Map a global record index to (file_index, local_index_within_file)."""
        total = self.cum_record_counts[-1] if self.cum_record_counts else 0
        if global_idx < 0 or global_idx >= total:
            raise IndexError(f"Record number {global_idx} out of range (0-{total - 1})")
        # bisect_right finds insertion point; subtract 1 for the owning file.
        file_idx = bisect.bisect_right(self.cum_record_counts, global_idx) - 1
        local_idx = global_idx - self.cum_record_counts[file_idx]
        return file_idx, local_idx
    
    def _build_line_positions_with_progress(self, filepath: Path) -> Tuple[int, List[int]]:
        """Build an index of byte positions for each line start."""
        file_size = filepath.stat().st_size
        line_positions = []
        line_count = 0
        
        print(f"Building line index for {file_size / (1024 * 1024):.1f} MB file...")
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Indexing records...", total=file_size)
                
                with open(filepath, 'rb') as f:
                    while True:
                        pos = f.tell()
                        line = f.readline()
                        if not line:
                            break
                        
                        line_positions.append(pos)
                        line_count += 1
                        
                        # Update progress more frequently for better time estimates
                        # Every 1000 lines for files < 1GB, every 5000 for larger files
                        update_interval = 1000 if file_size < 1024*1024*1024 else 5000
                        if line_count % update_interval == 0:
                            current_pos = f.tell()
                            progress.update(task, completed=current_pos)
                
                progress.update(task, completed=file_size)
        
        elif TQDM_AVAILABLE:
            with open(filepath, 'rb') as f:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc="Indexing records") as pbar:
                    last_update_pos = 0
                    while True:
                        pos = f.tell()
                        line = f.readline()
                        if not line:
                            break
                        
                        line_positions.append(pos)
                        line_count += 1
                        
                        # Update progress more frequently for better time estimates
                        update_interval = 1000 if file_size < 1024*1024*1024 else 5000
                        if line_count % update_interval == 0:
                            current_pos = f.tell()
                            pbar.update(current_pos - last_update_pos)
                            last_update_pos = current_pos
                    
                    # Final update to reach 100%
                    pbar.update(file_size - last_update_pos)
        
        else:
            print("Indexing records... (this may take a while for large files)")
            chunk_size = file_size // 20
            next_milestone = chunk_size
            
            with open(filepath, 'rb') as f:
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    
                    line_positions.append(pos)
                    line_count += 1

                    if line_count % 5000 == 0:
                        report_progress('build line index', pos, file_size,
                                        unit='bytes',
                                        note=f"{line_count:,} records")
                    if pos >= next_milestone:
                        percent = (pos / file_size) * 100
                        print(f"  {percent:.0f}% complete ({line_count:,} records indexed)...")
                        next_milestone += chunk_size
        
        return line_count, line_positions
    
    def _count_lines_with_progress(self, filepath: Path) -> int:
        """Count lines in a file with progress feedback (without building index)."""
        file_size = filepath.stat().st_size
        
        if file_size > 100 * 1024 * 1024:
            print(f"Large file detected ({file_size / (1024 * 1024):.1f} MB). Counting records...")
        
        line_count = 0
        bytes_read = 0
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Counting records...", total=file_size)
                
                with open(filepath, 'rb') as f:
                    for line in f:
                        line_count += 1
                        bytes_read += len(line)
                        
                        # Update every 500 lines for better time estimates
                        if line_count % 500 == 0:
                            progress.update(task, completed=bytes_read)
                
                progress.update(task, completed=file_size)
        
        elif TQDM_AVAILABLE:
            with open(filepath, 'rb') as f:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc="Counting records") as pbar:
                    last_update_bytes = 0
                    for line in f:
                        line_count += 1
                        bytes_read += len(line)
                        
                        # Update every 500 lines for better time estimates
                        if line_count % 500 == 0:
                            pbar.update(bytes_read - last_update_bytes)
                            last_update_bytes = bytes_read
                    
                    # Final update to reach 100%
                    pbar.update(file_size - last_update_bytes)
        
        else:
            print("Counting records... (this may take a while for large files)")
            chunk_size = file_size // 20
            next_milestone = chunk_size
            
            with open(filepath, 'rb') as f:
                for line in f:
                    line_count += 1
                    bytes_read += len(line)
                    
                    if bytes_read >= next_milestone:
                        percent = (bytes_read / file_size) * 100
                        print(f"  {percent:.0f}% complete ({line_count:,} records so far)...")
                        next_milestone += chunk_size
        
        return line_count
    
    def _estimate_line_count(self, filepath: Path, sample_size: int = 1000) -> tuple[int, bool]:
        """Estimate the line count by sampling the file."""
        file_size = filepath.stat().st_size
        
        print(f"Estimating record count from sample...")
        
        lines_read = 0
        bytes_read = 0
        
        with open(filepath, 'rb') as f:
            for _ in range(sample_size):
                line = f.readline()
                if not line:
                    return lines_read, True
                lines_read += 1
                bytes_read += len(line)
        
        avg_line_size = bytes_read / lines_read
        estimated_total = int(file_size / avg_line_size)
        
        return estimated_total, False
    
    def _npy_doc_count(self, path: Path) -> int:
        """Documents in a raw token shard, from BOS positions (mmap scan, no decode)."""
        starts, _ends = npy_doc_bounds(path, self._get_npy_tokenizer().bos_id)
        return len(starts) if self.npy_max_docs is None else min(len(starts), self.npy_max_docs)

    def _refresh_size_metadata(self):
        """(Re)compute the file-size fields from disk. Always, never from cache.

        These are a couple of stat() calls, so caching them saves nothing -- and it costs
        two ways: a decompressed temp file that gets regenerated makes the cached sizes
        wrong, and a cache written by a version that did not record them at all makes
        print_info raise KeyError on a key it assumes exists.
        """
        mb = 1024 * 1024
        if self.is_compressed:
            self.metadata['original_file_size'] = self.original_filepath.stat().st_size / mb
            self.metadata['decompressed_file_size'] = self.filepath.stat().st_size / mb
            self.metadata['file_size'] = self.metadata['decompressed_file_size']
            self.metadata['compression_ratio'] = (
                self.metadata['decompressed_file_size']
                / max(self.metadata['original_file_size'], 1e-9))
        elif self.is_json_array:
            self.metadata['original_file_size'] = self.original_filepath.stat().st_size / mb
            self.metadata['converted_file_size'] = self.filepath.stat().st_size / mb
            self.metadata['file_size'] = self.metadata['converted_file_size']
        else:
            self.metadata['file_size'] = self.filepath.stat().st_size / mb

        self.metadata['file_path'] = str(self.original_filepath.absolute())
        self.metadata['is_compressed'] = self.is_compressed
        self.metadata['is_json_array'] = self.is_json_array

    def _load_metadata(self):
        """Load basic metadata about the file."""
        if self.file_type == 'npy':
            self.metadata.update({
                'file_path': str(self.original_filepath.absolute()),
                'file_size': self.filepath.stat().st_size / (1024 * 1024),
                'num_rows': self._npy_doc_count(self.filepath),
                'columns': ['tokens'],
                'schema': {'tokens': 'token ids (raw shard)'},
                'num_columns': 1,
                'has_index': False,
                'is_token_shard': True,
                'is_compressed': False,
                'is_json_array': False,
            })
            return

        # Try to load from cache first
        if self.cache and not self.quick_mode:
            cache_data = self.cache.load()
            if cache_data:
                lp = cache_data.get('line_positions')
                if (self.file_type == 'jsonl' and lp
                        and not _line_index_matches(
                            self.filepath, lp,
                            cache_data['metadata'].get('num_rows'))):
                    print("Cached line index does not match the data file "
                          "(temp was replaced?); rebuilding metadata.")
                    cache_data = None
            if cache_data:
                self.metadata = cache_data['metadata']
                self.line_positions = cache_data.get('line_positions')

                # A cached metadata dict REPLACES self.metadata wholesale, so anything it
                # lacks is simply absent -- and print_info branches on the runtime
                # is_compressed flag, not on what the cache happens to contain. Recomputing
                # the size fields here keeps the two in step regardless of which version of
                # the tool wrote the cache.
                self._refresh_size_metadata()

                # Display cache info
                cached_time = cache_data.get('cached_at', 0)
                if cached_time:
                    age_hours = (time.time() - cached_time) / 3600
                    if self.console and RICH_AVAILABLE:
                        self.console.print(f"[green]Using cached metadata (age: {age_hours:.1f} hours)[/green]")
                    else:
                        print(f"Using cached metadata (age: {age_hours:.1f} hours)")

                return

        # No cache or cache invalid, load metadata normally
        self._refresh_size_metadata()

        if self.file_type == 'parquet':
            if self.console and RICH_AVAILABLE:
                self.console.print("[cyan]Loading parquet metadata...[/cyan]")
            elif not self.quick_mode:
                print("Loading parquet metadata...")
            
            parquet_file = pq.ParquetFile(self.filepath)
            self.metadata['num_rows'] = parquet_file.metadata.num_rows
            
            arrow_schema = parquet_file.schema_arrow
            self.metadata['num_columns'] = len(arrow_schema)
            self.metadata['columns'] = arrow_schema.names
            
            schema_dict = {}
            for i in range(len(arrow_schema)):
                field = arrow_schema.field(i)
                schema_dict[field.name] = str(field.type)
            self.metadata['schema'] = schema_dict
            
            # Save to cache
            if self.cache:
                self.cache.save(self.metadata)
        
        elif self.file_type == 'jsonl':
            # Get schema from first record
            with open(self.filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if first_line:
                    first_record = json.loads(first_line)
                    if not isinstance(first_record, dict):
                        raise ValueError(
                            f"Expected each record to be a JSON object, but got "
                            f"{type(first_record).__name__}. The file at "
                            f"{self.original_filepath} does not appear to contain "
                            f"object-shaped records."
                        )
                    self.metadata['columns'] = list(first_record.keys())
                    self.metadata['schema'] = {k: type(v).__name__ for k, v in first_record.items()}
                
                self.metadata['num_columns'] = len(self.metadata.get('columns', []))
            
            file_size_mb = self.metadata['file_size']
            
            if self.quick_mode:
                estimated_count, is_exact = self._estimate_line_count(self.filepath)
                self.metadata['num_rows'] = estimated_count
                self.metadata['count_is_estimate'] = not is_exact
                
                if not is_exact:
                    if self.console and RICH_AVAILABLE:
                        self.console.print(f"[yellow]Estimated ~{estimated_count:,} records (quick mode)[/yellow]")
                    else:
                        print(f"Estimated ~{estimated_count:,} records (quick mode)")
            
            elif file_size_mb > 500:
                # For very large files, ask about building index
                if self.console and RICH_AVAILABLE:
                    self.console.print(f"[yellow]Large JSONL file detected ({file_size_mb:.1f} MB)[/yellow]")
                else:
                    print(f"Large JSONL file detected ({file_size_mb:.1f} MB)")
                
                if self.non_interactive:
                    print("Building complete index (non-interactive mode)...")
                    response = '1'
                else:
                    print("Options:")
                    print("  1. Build complete index (enables fast random access, takes time)")
                    print("  2. Count only (faster, no random access)")
                    print("  3. Estimate (instant, approximate count)")
                    print("  4. Skip counting")

                    response = input("Choose option (1/2/3/4): ").strip()
                
                if response == '1':
                    # Build complete index
                    line_count, self.line_positions = self._build_line_positions_with_progress(self.filepath)
                    self.metadata['num_rows'] = line_count
                    self.metadata['count_is_estimate'] = False
                    self.metadata['has_index'] = True
                    print(f"Total records: {line_count:,}")
                    
                    # Save to cache with line positions
                    if self.cache:
                        self.cache.save(self.metadata, self.line_positions)
                
                elif response == '2':
                    # Count only
                    self.metadata['num_rows'] = self._count_lines_with_progress(self.filepath)
                    self.metadata['count_is_estimate'] = False
                    self.metadata['has_index'] = False
                    print(f"Total records: {self.metadata['num_rows']:,}")
                    
                    # Save to cache without line positions
                    if self.cache:
                        self.cache.save(self.metadata)
                
                elif response == '3':
                    # Estimate
                    estimated_count, is_exact = self._estimate_line_count(self.filepath)
                    self.metadata['num_rows'] = estimated_count
                    self.metadata['count_is_estimate'] = not is_exact
                    self.metadata['has_index'] = False
                    print(f"Estimated ~{estimated_count:,} records")
                
                else:
                    # Skip
                    self.metadata['num_rows'] = None
                    self.metadata['count_is_estimate'] = None
                    self.metadata['has_index'] = False
            
            else:
                # For smaller files, always build index
                if file_size_mb > 50:
                    print(f"Building index for {file_size_mb:.1f} MB file...")
                
                line_count, self.line_positions = self._build_line_positions_with_progress(self.filepath)
                self.metadata['num_rows'] = line_count
                self.metadata['count_is_estimate'] = False
                self.metadata['has_index'] = True
                
                # Save to cache with line positions
                if self.cache:
                    self.cache.save(self.metadata, self.line_positions)
    
    def get_record_by_position(self, index: int) -> Optional[Dict[str, Any]]:
        """Get a JSONL record using its cached byte position (O(1) random access).

        In directory mode, `index` is a GLOBAL record number that is mapped to the
        owning file via cumulative counts.
        """
        if self.is_directory:
            try:
                file_idx, local_idx = self._global_to_local(index)
            except IndexError:
                return None
            positions = self.file_line_positions[file_idx]
            if positions is None or local_idx >= len(positions):
                return None
            byte_pos = positions[local_idx]
            path = self.working_files[file_idx]
        else:
            if not self.line_positions or index >= len(self.line_positions):
                return None
            byte_pos = self.line_positions[index]
            path = self.filepath

        with open(path, 'rb') as f:
            f.seek(byte_pos)
            line = f.readline()
            if line:
                try:
                    return json.loads(line.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return None
        return None
    
    def sample_records(self, n: int = 5, random: bool = False) -> pd.DataFrame:
        """Sample n records from the dataset."""
        if self.file_type == 'npy':
            total = self.metadata.get('num_rows') or 0
            size = min(n, total)
            if random:
                import random as rand
                picks = sorted(rand.sample(range(total), size))
            else:
                picks = list(range(size))
            rows, idx = [], []
            for gi in picks:
                try:
                    rows.append(self.get_record(gi).iloc[0])
                    idx.append(gi)
                except Exception:
                    continue
            df = pd.DataFrame(rows)
            if idx:
                df.index = idx
            df._index_type = 'record_number'
            return df

        # Directory mode: route through global indices and reuse get_record per pick.
        if self.is_directory:
            total = self.cum_record_counts[-1]
            sample_size = min(n, total)
            if random:
                import random as rand
                picks = rand.sample(range(total), sample_size)
            else:
                picks = list(range(sample_size))
            rows = []
            indices_out = []
            for gi in picks:
                try:
                    rec_df = self.get_record(gi)
                    rows.append(rec_df.iloc[0])
                    indices_out.append(gi)
                except Exception:
                    continue
            df = pd.DataFrame(rows)
            if indices_out:
                df.index = indices_out
            df._index_type = 'record_number'
            return df

        if self.file_type == 'parquet':
            if random:
                df = pd.read_parquet(self.filepath)
                df = df.sample(n=min(n, len(df)))
                df._index_type = 'record_number'
                return df
            else:
                df = pd.read_parquet(self.filepath).head(n)
                df._index_type = 'record_number'
                return df

        elif self.file_type == 'jsonl':
            records = []
            indices = []
            
            # Use cached line positions for fast random access if available
            if random and self.line_positions:
                import random as rand
                
                print(f"Fast random sampling {n} records using index...")
                total_records = len(self.line_positions)
                sample_size = min(n, total_records)
                
                # Random sample of indices
                sampled_indices = rand.sample(range(total_records), sample_size)
                
                for idx in sampled_indices:
                    record = self.get_record_by_position(idx)
                    if record:
                        records.append(record)
                        indices.append(idx)
                
                print(f"Successfully sampled {len(records)} records")
                
                # Create DataFrame with record numbers
                df = pd.DataFrame(records)
                if indices:
                    df.index = indices
                df._index_type = 'record_number'  # These are record numbers!
                return df
            
            elif random:
                # Fallback to byte-seeking method if no index
                import random as rand
                
                print(f"Random sampling {n} records (no index available)...")
                
                file_size = self.filepath.stat().st_size
                
                if file_size < 1000:
                    with open(self.filepath, 'r', encoding='utf-8') as f:
                        all_records = [json.loads(line) for line in f]
                        if all_records:
                            sample_size = min(n, len(all_records))
                            sampled = rand.sample(list(enumerate(all_records)), sample_size)
                            indices = [i for i, _ in sampled]
                            records = [r for _, r in sampled]
                            
                            df = pd.DataFrame(records)
                            if indices:
                                df.index = indices
                            df._index_type = 'record_number'
                            return df
                else:
                    attempts = 0
                    max_attempts = min(n * 50, 500)
                    seen_positions = set()
                    
                    with open(self.filepath, 'rb') as f:
                        while len(records) < n and attempts < max_attempts:
                            attempts += 1
                            
                            max_pos = max(0, file_size - 100)
                            random_pos = rand.randint(0, max_pos)
                            
                            if seen_positions and min(abs(random_pos - p) for p in seen_positions) < 5000:
                                continue
                            
                            seen_positions.add(random_pos)
                            
                            f.seek(random_pos)
                            
                            if random_pos > 0:
                                f.readline()
                            
                            actual_pos = f.tell()
                            line = f.readline()
                            
                            if line and len(line) > 2:
                                try:
                                    line_str = line.decode('utf-8').strip()
                                    if line_str:
                                        record = json.loads(line_str)
                                        records.append(record)
                                        indices.append(actual_pos)
                                        
                                        if len(records) <= 5 or len(records) % 10 == 0:
                                            print(f"  Found {len(records)}/{n} records...")
                                
                                except (json.JSONDecodeError, UnicodeDecodeError):
                                    pass
                    
                    if len(records) < n:
                        print(f"Found {len(records)} valid records")
                    
                    # Create DataFrame with byte positions
                    df = pd.DataFrame(records)
                    if indices:
                        df.index = indices
                    df._index_type = 'byte_position'  # These are byte positions!
                    return df
            
            else:
                # Sequential reading
                if self.line_positions:
                    # Use index for sequential access too
                    for i in range(min(n, len(self.line_positions))):
                        record = self.get_record_by_position(i)
                        if record:
                            records.append(record)
                            indices.append(i)
                else:
                    # Fallback to regular sequential reading
                    with open(self.filepath, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if i >= n:
                                break
                            try:
                                record = json.loads(line.strip())
                                records.append(record)
                                indices.append(i)
                            except json.JSONDecodeError:
                                pass
                
                df = pd.DataFrame(records)
                if indices:
                    df.index = indices
                df._index_type = 'record_number'  # Sequential reads are record numbers
                return df
    
    _NPY_BOUNDS_CACHE_MAX = 8

    def _npy_spans_cached(self, file_idx: int, path: Path):
        """BOS-inclusive document spans for one shard, cached.

        Inclusive rather than content-only because these spans TILE the shard with no
        gaps, which is what lets a token offset be mapped back to its document with a
        single searchsorted. Scanning a shard for BOS reads the whole file, so an
        uncached lookup per record would make inspecting a cluster O(shards).
        """
        hit = self._npy_bounds_cache.get(file_idx)
        if hit is None:
            hit = npy_doc_spans(path, self._get_npy_tokenizer().bos_id)
            if len(self._npy_bounds_cache) >= self._NPY_BOUNDS_CACHE_MAX:
                self._npy_bounds_cache.pop(next(iter(self._npy_bounds_cache)))
            self._npy_bounds_cache[file_idx] = hit
        return hit

    def _get_npy_record(self, index: int) -> pd.DataFrame:
        """Decode ONE document from a raw token shard, on demand.

        Raw-shard mode exists to avoid decoding the corpus up front -- on a 1647-shard
        tree that is days of work. But a duplicate you cannot read is not much use, so
        single-record decoding is done lazily here: one document's token span, one
        decode call, milliseconds.
        """
        if self.is_directory:
            file_idx, local_idx = self._global_to_local(index)
            path = self.working_files[file_idx]
        else:
            file_idx, local_idx, path = 0, index, self.filepath

        starts, ends = self._npy_spans_cached(file_idx, path)
        if local_idx >= len(starts):
            raise ValueError(f"Record {index} out of range for {path.name}")

        arr = np.load(path, mmap_mode='r')
        s, e = int(starts[local_idx]), int(ends[local_idx])
        if s < len(arr) and int(arr[s]) == self._get_npy_tokenizer().bos_id:
            s += 1                                   # spans are BOS-inclusive; skip it
        ids = np.asarray(arr[s:e]).tolist()
        text = self._get_npy_tokenizer().decode(ids)
        df = pd.DataFrame([{'tokens': len(ids), 'text': text}], index=[index])
        df._index_type = 'record_number'
        return df

    def get_record(self, index: int) -> pd.DataFrame:
        """Get a specific record by (global) index."""
        if self.file_type == 'npy':
            if self.metadata.get('num_rows') is not None and not (
                    0 <= index < self.metadata['num_rows']):
                raise ValueError(
                    f"Record number {index} out of range. Dataset has "
                    f"{self.metadata['num_rows']} records (0-{self.metadata['num_rows']-1})")
            return self._get_npy_record(index)
        if self.metadata.get('num_rows') is not None:
            if index < 0 or index >= self.metadata['num_rows']:
                raise ValueError(
                    f"Record number {index} out of range. Dataset has "
                    f"{self.metadata['num_rows']} records (0-{self.metadata['num_rows']-1})"
                )
        elif index < 0:
            raise ValueError(f"Record number must be non-negative (got {index})")

        # Resolve to a working file + local index. Single-file mode is just file 0.
        if self.is_directory:
            file_idx, local_idx = self._global_to_local(index)
            working_path = self.working_files[file_idx]
        else:
            file_idx = 0
            local_idx = index
            working_path = self.filepath

        if self.file_type == 'parquet':
            parquet_file = pq.ParquetFile(working_path)

            current_idx = 0
            for i in range(parquet_file.num_row_groups):
                row_group = parquet_file.metadata.row_group(i)
                group_rows = row_group.num_rows

                if current_idx <= local_idx < current_idx + group_rows:
                    df = parquet_file.read_row_group(i).to_pandas()
                    inner_idx = local_idx - current_idx
                    result = df.iloc[[inner_idx]]
                    result.index = [index]
                    result._index_type = 'record_number'
                    return result

                current_idx += group_rows

            df = pd.read_parquet(working_path)
            result = df.iloc[[local_idx]]
            result.index = [index]
            result._index_type = 'record_number'
            return result

        elif self.file_type == 'jsonl':
            # Use index for O(1) access if available
            if self.is_directory:
                positions = self.file_line_positions[file_idx]
            else:
                positions = self.line_positions

            if positions:
                record = self.get_record_by_position(index)
                if record:
                    df = pd.DataFrame([record], index=[index])
                    df._index_type = 'record_number'
                    return df
                raise ValueError(f"Could not read record at index {index}")

            # Fallback to sequential reading (single-file mode without index)
            with open(working_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == local_idx:
                        record = json.loads(line)
                        df = pd.DataFrame([record], index=[index])
                        df._index_type = 'record_number'
                        return df

            raise ValueError(f"Could not find record at number {index}")
    
    def _resolve_field(self, name: str) -> Optional[str]:
        """Return the actual column name matching `name` case-insensitively, or None."""
        cols = self.metadata.get('columns', []) or []
        if name in cols:
            return name
        name_lc = name.lower()
        for col in cols:
            if col.lower() == name_lc:
                return col
        return None

    def _encode_query_variants(self, query: str) -> List[Tuple[str, np.ndarray]]:
        """Token-id forms of `query` to search for.

        BPE is context-sensitive: "the" mid-sentence encodes as a single " the" token,
        while at the start of a document it encodes as "the". A single encoding therefore
        misses most real occurrences, so we search a small set of surface forms and report
        which ones hit. Leading/trailing special tokens are stripped -- some tokenizers add
        BOS on encode, and that would never appear mid-document.
        """
        tok = self._get_npy_tokenizer()
        bos = getattr(tok, 'bos_id', None)
        eos = getattr(tok, 'eos_id', None)

        forms = [query, ' ' + query]
        if query[:1].islower():                       # sentence-initial capitalisation
            cap = query[0].upper() + query[1:]
            forms += [cap, ' ' + cap]

        out: List[Tuple[str, np.ndarray]] = []
        seen = set()
        for form in forms:
            try:
                ids = list(tok.encode(form))
            except Exception as e:
                raise ValueError(f"Could not tokenize query {form!r}: {e}")
            while ids and bos is not None and ids[0] == bos:
                ids.pop(0)
            while ids and eos is not None and ids[-1] == eos:
                ids.pop()
            key = tuple(ids)
            if ids and key not in seen:
                seen.add(key)
                out.append((form, np.asarray(ids, dtype=np.int64)))
        if not out:
            raise ValueError(f"Query {query!r} produced no tokens")
        return out

    @staticmethod
    def _token_subsequence_positions(arr: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Start offsets where token sequence `q` occurs in `arr`.

        Anchors on the query's HIGHEST token id rather than its first. BPE assigns low ids
        to the merges it learned earliest, which are the most frequent ones, so the highest
        id is a good cheap proxy for the rarest token -- and anchoring on the rarest token
        is what keeps the candidate set small for queries that begin with a common word.
        """
        m, n = int(q.size), int(arr.size)
        if m == 0 or n < m:
            return np.empty(0, dtype=np.int64)
        anchor = int(np.argmax(q))
        window = np.asarray(arr[anchor:n - m + 1 + anchor])
        cand = np.flatnonzero(window == q[anchor]).astype(np.int64)
        for j in range(m):
            if j == anchor or cand.size == 0:
                continue
            cand = cand[np.asarray(arr[cand + j]) == q[j]]
        return cand

    def _find_all_records_npy(self, terms: List[str], limit: Optional[int],
                              match_all: bool) -> List[int]:
        """findall over RAW token shards -- searches token ids, decodes nothing.

        Full coverage of every shard with no decode and no disk cost. The tradeoff versus
        text search is honest and worth stating: matching is case- and whitespace-sensitive
        at the token level, so recall depends on the surface forms tried by
        _encode_query_variants, and substring matches inside a larger word are not found.
        """
        variants = {t: self._encode_query_variants(t) for t in terms}
        if self.console and RICH_AVAILABLE:
            for t in terms:
                forms = ", ".join(f"{f!r}({len(ids)}tok)" for f, ids in variants[t])
                self.console.print(f"[dim]  {t!r} -> {forms}[/dim]")

        srcs = self.working_files if self.is_directory else [self.filepath]
        per_term: List[set] = [set() for _ in terms]
        hits_per_term = [0] * len(terms)

        def scan(progress=None, task=None):
            for fidx, path in enumerate(srcs):
                arr = np.load(path, mmap_mode='r')
                starts, _ends = self._npy_spans_cached(fidx, path)
                base = self.cum_record_counts[fidx] if self.is_directory else 0
                for ti, term in enumerate(terms):
                    for _form, q in variants[term]:
                        pos = self._token_subsequence_positions(arr, q)
                        if pos.size:
                            docs = np.searchsorted(starts, pos, side='right') - 1
                            found = {base + int(d) for d in np.unique(docs)}
                            hits_per_term[ti] += len(found - per_term[ti])
                            per_term[ti] |= found
                if progress is not None:
                    matched = (set.intersection(*per_term) if match_all and len(terms) > 1
                               else set().union(*per_term))
                    progress.update(task, completed=fidx + 1, matches=len(matched))
                if limit is not None and not match_all:
                    if len(set().union(*per_term)) >= limit:
                        return

        if RICH_AVAILABLE and self.console:
            with Progress(SpinnerColumn(),
                          TextColumn("[progress.description]{task.description}"),
                          BarColumn(),
                          TextColumn("{task.completed:,}/{task.total:,} shards"),
                          TextColumn("matches: {task.fields[matches]:,}"),
                          TimeRemainingColumn(), console=self.console) as progress:
                task = progress.add_task("Token search...", total=len(srcs), matches=0)
                scan(progress, task)
        else:
            print(f"Token search across {len(srcs)} shard(s)...")
            scan()

        if match_all and len(terms) > 1:
            matched = set.intersection(*per_term)
        else:
            matched = set().union(*per_term)
        if len(terms) > 1:
            print("  per-term docs: " +
                  "  ".join(f"{terms[i][:12]}:{len(per_term[i]):,}" for i in range(len(terms))))
        out = sorted(matched)
        return out[:limit] if limit is not None else out

    def find_all_records(self, query: Union[str, List[str]],
                         field: Optional[str] = None,
                         regex: bool = False,
                         limit: Optional[int] = None,
                         count_terms: bool = False) -> List[int]:
        """Scan the dataset and return record indices for matches.

        `query` may be a single string (one term) or a list of strings (AND match;
        every term must be present in the record/field). With limit=None, scans
        every file. With limit set, short-circuits as soon as `limit` matches are
        collected. The caller can detect a limit hit via `len(indices) == limit`.

        In directory mode, returned indices are GLOBAL across the directory; rows
        from each file are scanned in alphabetical order with offsets applied.
        """
        indices: List[int] = []

        terms: List[str] = [query] if isinstance(query, str) else list(query)
        if not terms or not all(t for t in terms):
            raise ValueError("query must be a non-empty string or list of non-empty strings")

        if self.file_type == 'npy':
            if regex:
                raise ValueError("Regex search needs decoded text; raw-shard mode searches "
                                 "token sequences. Drop -r, or reopen without --raw-shards.")
            if field:
                raise ValueError("A token shard has no fields to scope a search to.")
            return self._find_all_records_npy(terms, limit, match_all=len(terms) > 1)

        if regex:
            try:
                for t in terms:
                    re.compile(t)
            except re.error as e:
                raise ValueError(f"Invalid regex: {e}")

        # Per-file iterator: (file_idx, working_path, base_offset, total_rows_or_None)
        if self.is_directory:
            file_iter = [
                (i, self.working_files[i], self.cum_record_counts[i], self.file_record_counts[i])
                for i in range(len(self.working_files))
            ]
        else:
            file_iter = [(0, self.filepath, 0, self.metadata.get('num_rows'))]

        if self.file_type == 'parquet':
            total_rows = self.metadata.get('num_rows') or 0

            def _term_mask(df: pd.DataFrame, term: str) -> pd.Series:
                if field:
                    if field in df.columns:
                        return df[field].astype(str).str.contains(
                            term, case=False, na=False, regex=regex
                        )
                    return pd.Series([False] * len(df), index=df.index)
                m = pd.Series([False] * len(df), index=df.index)
                for col in df.select_dtypes(include=['object', 'str']).columns:
                    m = m | df[col].astype(str).str.contains(
                        term, case=False, na=False, regex=regex
                    )
                for col in df.select_dtypes(include=['number']).columns:
                    m = m | df[col].astype(str).str.contains(
                        term, case=False, na=False, regex=False
                    )
                return m

            scanned = 0

            def scan_all(progress=None, task=None):
                nonlocal scanned
                for fidx, working_path, base_offset, _ in file_iter:
                    pf = pq.ParquetFile(working_path)
                    file_cursor = base_offset
                    for batch in pf.iter_batches(batch_size=10000):
                        df = batch.to_pandas()
                        df.index = range(file_cursor, file_cursor + len(df))

                        mask = pd.Series([True] * len(df), index=df.index)
                        for t in terms:
                            mask = mask & _term_mask(df, t)

                        indices.extend(df.index[mask].tolist())
                        file_cursor += len(df)
                        scanned += len(df)
                        report_progress('scan records', scanned, total_rows,
                                        note=f"{len(indices):,} matches")
                        if progress is not None and task is not None:
                            progress.update(task, completed=scanned, matches=len(indices))
                        if limit is not None and len(indices) >= limit:
                            del indices[limit:]
                            return

            if RICH_AVAILABLE and self.console and total_rows:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed:,}/{task.total:,}"),
                    TextColumn("matches: {task.fields[matches]:,}"),
                    TimeRemainingColumn(),
                    console=self.console,
                ) as progress:
                    task = progress.add_task("Scanning records...", total=total_rows, matches=0)
                    scan_all(progress, task)
                    progress.update(task, completed=total_rows, matches=len(indices))
            else:
                print("Scanning records...")
                scan_all()
                print(f"  Scanned {scanned:,} records, {len(indices):,} matches")

            return indices

        elif self.file_type == 'jsonl':
            patterns = [re.compile(t, re.IGNORECASE) for t in terms] if regex else None
            terms_lc = [t.lower() for t in terms]

            def _term_in_record(record: Dict[str, Any], idx: int) -> bool:
                if field:
                    if field not in record:
                        return False
                    val = str(record[field])
                    if regex:
                        return bool(patterns[idx].search(val))
                    return terms_lc[idx] in val.lower()
                for v in record.values():
                    v_str = str(v)
                    if regex:
                        if patterns[idx].search(v_str):
                            return True
                    elif terms_lc[idx] in v_str.lower():
                        return True
                return False

            # Per-term hit tally (live diagnostic for -a AND searches: shows which term is the
            # zero that's killing matches). Always shown for multi-term; the counts come free
            # from the term-checks we already do. IMPORTANT: in default (short-circuit) mode the
            # tally is a LOWER BOUND -- once a term fails we stop checking the rest, so later
            # terms are undercounted. A zero is still meaningful (that term never matched where
            # earlier terms did). count_terms=True disables short-circuit for EXACT counts (slower
            # on big scans, since every term is checked on every record).
            multi_term = len(terms) > 1
            term_hits = [0] * len(terms)
            hits_label = "hits" if count_terms else "hits~"   # ~ = approximate (short-circuited)

            def _fmt_hits() -> str:
                def lab(t: str) -> str:
                    return (t[:12] + '…') if len(t) > 13 else t
                return "  ".join(f"{lab(terms[i])}:{term_hits[i]:,}" for i in range(len(terms)))

            def line_matches(record: Dict[str, Any]) -> bool:
                if not multi_term:
                    return _term_in_record(record, 0)
                if count_terms:                                     # exact: check every term
                    present = [_term_in_record(record, i) for i in range(len(terms))]
                    for i, p in enumerate(present):
                        if p:
                            term_hits[i] += 1
                    return all(present)
                for i in range(len(terms)):                         # fast: short-circuit + tally
                    if _term_in_record(record, i):
                        term_hits[i] += 1
                    else:
                        return False
                return True

            total_size = sum(p.stat().st_size for _, p, _, _ in file_iter)
            scanned_bytes = 0

            class _LimitHit(Exception):
                pass

            def scan_one(working_path: Path, base_offset: int,
                         progress=None, task=None, live=None, render=None) -> int:
                nonlocal scanned_bytes
                bytes_read_local = 0
                line_num = 0
                update_every = 2000
                with open(working_path, 'rb') as f:
                    while True:
                        line_bytes = f.readline()
                        if not line_bytes:
                            break
                        bytes_read_local += len(line_bytes)
                        scanned_bytes += len(line_bytes)
                        try:
                            record = json.loads(line_bytes)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            line_num += 1
                            continue
                        if line_matches(record):
                            indices.append(base_offset + line_num)
                            if limit is not None and len(indices) >= limit:
                                if progress is not None and task is not None:
                                    progress.update(task, completed=scanned_bytes, matches=len(indices))
                                raise _LimitHit
                        line_num += 1
                        if line_num % update_every == 0:
                            # The CLI's live feedback (match count; per-term tallies
                            # for AND searches), carried as a progress note so the
                            # web view gets it too.
                            report_progress(
                                'scan records', scanned_bytes, total_size,
                                unit='bytes',
                                note=(f"{hits_label}: {_fmt_hits()}" if multi_term
                                      else f"{len(indices):,} matches"))
                            if progress is not None and task is not None:
                                progress.update(task, completed=scanned_bytes, matches=len(indices))
                                if live is not None and render is not None:
                                    live.update(render())
                            elif multi_term:
                                print(f"  {hits_label}: {_fmt_hits()}   ", end='\r', flush=True)
                return line_num

            if RICH_AVAILABLE and self.console and total_size:
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TextColumn("matches: {task.fields[matches]:,}"),
                    TextColumn("file: {task.fields[file_label]}"),
                    TimeRemainingColumn(),
                    console=self.console,
                )
                task = progress.add_task("Scanning records...", total=total_size,
                                         matches=0, file_label="")

                def _scan_loop(live=None, render=None):
                    try:
                        for fidx, working_path, base_offset, _ in file_iter:
                            label = (self.source_files[fidx].name if self.is_directory
                                     else working_path.name)
                            progress.update(task, file_label=f"[{fidx + 1}/{len(file_iter)}] {label}")
                            scan_one(working_path, base_offset, progress, task, live, render)
                    except _LimitHit:
                        pass
                    progress.update(task, completed=total_size, matches=len(indices))

                if multi_term:
                    # Tally on its own line ABOVE the progress bar (a bar column would overflow
                    # the terminal width and squeeze the bar out). Live drives both lines.
                    from rich.live import Live
                    from rich.console import Group
                    from rich.text import Text

                    def _render():
                        return Group(Text(f"{hits_label}: {_fmt_hits()}", style="cyan"), progress)

                    with Live(_render(), console=self.console, refresh_per_second=12) as live:
                        _scan_loop(live, _render)
                        live.update(_render())
                else:
                    with progress:
                        _scan_loop()
            else:
                print("Scanning records...")
                try:
                    for fidx, working_path, base_offset, _ in file_iter:
                        if self.is_directory:
                            print(f"  [{fidx + 1}/{len(file_iter)}] {self.source_files[fidx].name}")
                        scan_one(working_path, base_offset)
                except _LimitHit:
                    pass
                if multi_term:
                    print(f"  {hits_label}: {_fmt_hits()}   ")   # final tally, clears the \r status line
                print(f"  Done. {len(indices):,} matches.")

            return indices

        return indices

    # ---- Near-duplicate detection -------------------------------------------

    TEXT_FIELD_CANDIDATES = ('text', 'content', 'message', 'body', 'document',
                             'raw_content', 'article')
    _WORD_RE = re.compile(r"\S+")

    def _npy_sources(self) -> Optional[List[Path]]:
        """Original .npy shard paths if this dataset is pre-tokenized, else None."""
        srcs = self.source_files if self.is_directory else [self.original_filepath]
        if srcs and all(p.suffix.lower() == '.npy' for p in srcs):
            return list(srcs)
        return None

    def _resolve_text_field(self, field: Optional[str] = None) -> str:
        if field:
            resolved = self._resolve_field(field)
            if resolved is None:
                raise ValueError(f"Field '{field}' not found. Available: "
                                 f"{', '.join(self.metadata.get('columns') or [])}")
            return resolved
        for candidate in self.TEXT_FIELD_CANDIDATES:
            resolved = self._resolve_field(candidate)
            if resolved:
                return resolved
        cols = self.metadata.get('columns') or []
        if cols:
            return cols[0]
        raise ValueError("No text field found; pass -f <field>")

    def _text_to_ids(self, text: Any) -> np.ndarray:
        """Map text to an integer id sequence for shingling.

        Shingling only cares about token IDENTITY, never token meaning, so hashing
        whitespace words to ids is equivalent to a real vocabulary for this purpose and
        needs no model. When a tokenizer IS configured we use it instead, so that text
        datasets shingle on the same units as the .npy shards.
        """
        if text is None:
            return np.empty(0, dtype=np.int32)
        s = str(text).lower()
        if self.tok_kind and self.tok_path:
            return np.asarray(self._get_npy_tokenizer().encode(s), dtype=np.int32)
        memo = self._word_id_memo
        if len(memo) > 2_000_000:
            # Pure cache -- ids are content-derived (blake2b of the word), so clearing
            # never changes any signature. Left unbounded, a large corpus fills it with
            # every distinct typo and name (tens of millions of entries), and gen-2 GC
            # passes over that dict grow until sketch throughput visibly decays.
            memo.clear()
        out = []
        for w in self._WORD_RE.findall(s):
            v = memo.get(w)
            if v is None:
                v = int.from_bytes(
                    hashlib.blake2b(w.encode('utf-8', 'replace'), digest_size=4).digest(),
                    'little') & 0x7FFFFFFF
                memo[w] = v
            out.append(v)
        return np.asarray(out, dtype=np.int32)

    def _iter_records_streaming(self):
        """Yield (global_index, record) in global record order, without an index."""
        if self.is_directory:
            files = [(self.working_files[i], self.cum_record_counts[i])
                     for i in range(len(self.working_files))]
        else:
            files = [(self.filepath, 0)]

        if self.file_type == 'jsonl':
            for path, base in files:
                line_no = 0
                with open(path, 'rb') as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            line_no += 1
                            continue
                        yield base + line_no, rec
                        line_no += 1
        else:
            for path, base in files:
                pf = pq.ParquetFile(path)
                cursor = base
                for batch in pf.iter_batches(batch_size=2000):
                    df = batch.to_pandas()
                    for offset, (_idx, row) in enumerate(df.iterrows()):
                        yield cursor + offset, row.to_dict()
                    cursor += len(df)

    def _iter_dedup_docs(self, field: Optional[str] = None, sample: Optional[int] = None,
                         skip: int = 0):
        """Yield token-id arrays in GLOBAL record order, so cluster members are record #s.

        For .npy sources this reads the shard directly -- the ordering matches
        convert_npy_to_jsonl (both split on BOS identically), so record numbers stay
        consistent whether or not the dataset was decoded.

        `skip` (sketch resume) fast-forwards past already-sketched documents
        WITHOUT tokenizing them -- text->ids is the expensive step, so skipping
        costs only the record iteration.
        """
        npys = self._npy_sources()
        n_seen = 0
        if npys:
            bos = self._get_npy_tokenizer().bos_id
            for path in npys:
                for _local, ids in iter_npy_token_docs(path, bos, self.npy_max_docs):
                    if sample and n_seen >= sample:
                        return
                    if n_seen >= skip:
                        yield ids
                    n_seen += 1
            return

        text_field = self._resolve_text_field(field)
        for _gi, rec in self._iter_records_streaming():
            if sample and n_seen >= sample:
                return
            if n_seen >= skip:
                yield self._text_to_ids(rec.get(text_field))
            n_seen += 1

    # ---- Named result sets --------------------------------------------------

    def _sets_path(self) -> Path:
        base = self.original_filepath
        cache_dir = (base if base.is_dir() else base.parent) / '.dataset_explorer_cache'
        cache_dir.mkdir(exist_ok=True)
        tag = hashlib.md5(str(base.absolute()).encode()).hexdigest()[:8]
        stem = base.name if base.is_dir() else base.stem
        return cache_dir / f"{stem}_{tag}.sets.gz"

    def _load_sets(self):
        """Load persisted result sets. Curating millions of records spans sessions, so a
        set that vanishes when the REPL closes is not much use.

        Sets are keyed to a record count: if the dataset has changed, every stored index
        may now point at a different document, so they are dropped rather than silently
        applied to the wrong records.
        """
        self.result_sets: Dict[str, Dict[str, Any]] = {}
        path = self._sets_path()
        if not path.exists():
            return
        try:
            with gzip.open(path, 'rb') as f:
                blob = pickle.load(f)
        except Exception as e:
            print(f"Could not read saved result sets ({e}); starting empty.")
            return
        n_now = self.metadata.get('num_rows')
        if blob.get('n_docs') is not None and n_now is not None and blob['n_docs'] != n_now:
            print(f"Saved result sets were built against {blob['n_docs']:,} records but this "
                  f"dataset has {n_now:,}. Discarding them -- record numbers would not line "
                  f"up.")
            return
        self.result_sets = blob.get('sets', {})
        if self.result_sets:
            total = len(self.result_sets)
            print(f"Loaded {total} saved result set{'s' if total != 1 else ''}: "
                  f"{', '.join(sorted(self.result_sets)[:6])}"
                  f"{' ...' if total > 6 else ''}  ('sets' to list)")

    def _save_sets(self):
        try:
            with gzip.open(self._sets_path(), 'wb', compresslevel=6) as f:
                pickle.dump({'version': 1, 'n_docs': self.metadata.get('num_rows'),
                             'sets': self.result_sets}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Could not save result sets: {e}")

    def _auto_dupe_set_name(self) -> str:
        i = 1
        while f"dupes{i:02d}" in self.result_sets:
            i += 1
        return f"dupes{i:02d}"

    def _auto_set_name(self) -> str:
        i = 1
        while f"search{i:02d}" in self.result_sets:
            i += 1
        return f"search{i:02d}"

    def store_result_set(self, indices: Sequence[int], query: str, kind: str,
                         name: Optional[str] = None, activate: bool = True) -> str:
        """Record a search result under a name and make it the active set."""
        if name is None:
            name = self._auto_set_name()
        # int32 scales both ways: 20 bytes for a handful of hits, 32 MB for 8M of them,
        # and it compresses well on disk because the indices are sorted.
        arr = np.asarray(sorted(indices), dtype=np.int32)
        self.result_sets[name] = {
            'indices': arr, 'query': query, 'kind': kind,
            'created': time.time(), 'count': int(arr.size),
        }
        self._save_sets()
        if activate:
            self.activate_result_set(name, announce=False)
        return name

    def activate_result_set(self, name: str, announce: bool = True) -> bool:
        entry = self.result_sets.get(name)
        if entry is None:
            near = [k for k in self.result_sets if name.lower() in k.lower()][:5]
            print(f"No result set named {name!r}."
                  + (f" Did you mean: {', '.join(near)}?" if near else " Try 'sets'."))
            return False
        self.search_state = {
            'indices': entry['indices'].tolist(), 'cursor': 0,
            'query': entry['query'], 'field': None, 'regex': False,
            'limit_hit': False, 'set_name': name,
        }
        if announce:
            print(f"Switched to {name!r}: {entry['count']:,} records "
                  f"({entry['kind']})  {entry['query']!r}")
        return True

    def display_result_sets(self):
        if not self.result_sets:
            print("No result sets yet. Any findall stores one; name it with -n <name>.")
            return
        total = self.metadata.get('num_rows') or 0
        active = (self.search_state or {}).get('set_name')
        rows = sorted(self.result_sets.items(), key=lambda kv: -kv[1]['created'])
        if RICH_AVAILABLE and self.console:
            from rich.markup import escape
            tbl = Table(title=f"Result sets ({total:,} records in dataset)")
            tbl.add_column("", style="green")
            tbl.add_column("Name", style="cyan")
            tbl.add_column("Records", style="green", justify="right")
            tbl.add_column("%", style="green", justify="right")
            tbl.add_column("Kind", style="magenta")
            tbl.add_column("Query", style="white")
            for name, e in rows:
                tbl.add_row("*" if name == active else "", name, f"{e['count']:,}",
                            f"{e['count'] / max(total, 1) * 100:.2f}", e['kind'],
                            escape(e['query'][:70]))
            self.console.print(tbl)
        else:
            for name, e in rows:
                mark = '*' if name == active else ' '
                print(f" {mark} {name:<20} {e['count']:>10,}  "
                      f"{e['count'] / max(total, 1) * 100:>6.2f}%  {e['kind']:<10} "
                      f"{e['query'][:60]}")
        print("'switch <name>' to activate, 'delete <name>' to remove, "
              "'export <dir> --include/--exclude <names>' to write.")

    def delete_result_set(self, name: str) -> bool:
        if name not in self.result_sets:
            print(f"No result set named {name!r}.")
            return False
        del self.result_sets[name]
        self._save_sets()
        if (self.search_state or {}).get('set_name') == name:
            self.search_state = None
        print(f"Deleted result set {name!r}.")
        return True

    def resolve_set_names(self, spec: str) -> List[str]:
        """Parse a comma-separated list of set names, erroring on any unknown one."""
        names = [n.strip() for n in spec.split(',') if n.strip()]
        missing = [n for n in names if n not in self.result_sets]
        if missing:
            raise ValueError(f"Unknown result set(s): {', '.join(missing)}. "
                             f"Known: {', '.join(sorted(self.result_sets)) or '(none)'}")
        return names

    # ---- Metadata index & query --------------------------------------------

    def _metaindex_path(self) -> Path:
        base = self.original_filepath
        cache_dir = (base if base.is_dir() else base.parent) / '.dataset_explorer_cache'
        cache_dir.mkdir(exist_ok=True)
        tag = hashlib.md5(str(base.absolute()).encode()).hexdigest()[:8]
        stem = base.name if base.is_dir() else base.stem
        return cache_dir / f"{stem}_{tag}.metaindex.parquet"

    def _metaindex_sources(self) -> List[Path]:
        """Source files in GLOBAL RECORD ORDER -- the same order _iter_records_streaming
        walks. Index row i must be record i, or every query result points at the wrong
        document."""
        if self.is_directory:
            return list(self.source_files)
        return [self.original_filepath]

    def get_metaindex(self, rebuild: bool = False):
        """Load the metadata index, building it on first use."""
        if not METAQUERY_AVAILABLE:
            raise RuntimeError("metaquery.py and metaindex.py must sit beside "
                               "dataset_explorer.py")
        if getattr(self, '_metaindex', None) is not None and not rebuild:
            return self._metaindex

        path = self._metaindex_path()
        if rebuild or not path.exists():
            srcs = self._metaindex_sources()
            total = self.metadata.get('num_rows') or 0
            print(f"Building metadata index over {len(srcs)} file(s)"
                  f"{f', {total:,} records' if total else ''}...")
            hint = set()
            if self.text_field:
                try:
                    hint.add(self._resolve_text_field(self.text_field))
                except Exception:
                    hint.add(self.text_field)
            print("  (metadata only -- " +
                  (f"body field {sorted(hint)[0]!r} given via --field" if hint
                   else "body text detected by value length") +
                  ", never indexed)")
            if RICH_AVAILABLE and self.console and total:
                with Progress(SpinnerColumn(),
                              TextColumn("[progress.description]{task.description}"),
                              BarColumn(),
                              TextColumn("{task.completed:,}/{task.total:,}"),
                              TimeRemainingColumn(), console=self.console) as progress:
                    task = progress.add_task("Indexing metadata...", total=total)
                    stats = metaindex.build_index(
                        srcs, path, text_fields=hint,
                        progress=lambda n: progress.update(
                            task, completed=min(n, total)))
            else:
                n_disc = min(metaindex.DISCOVER_RECORDS, total) if total \
                    else metaindex.DISCOVER_RECORDS
                print(f"  discovering fields (adaptive, up to {n_disc:,} records)...")
                stats = metaindex.build_index(
                    srcs, path, text_fields=hint,
                    progress=lambda n: report_progress(
                        'index metadata', min(n, total) if total else n, total),
                    discover_progress=lambda n: report_progress(
                        'discover metadata fields', min(n, n_disc), n_disc))
            print(f"  {stats['rows']:,} records, {len(stats['fields'])} fields, "
                  f"{stats['size_mb']:.1f} MB"
                  + (f" (schema stabilized after {stats['discover_scanned']:,} records)"
                     if stats.get('discover_scanned') else ""))
            for f, avg in sorted(stats.get('body_fields', {}).items()):
                print(f"  excluded {f!r} as body text (mean {avg:,} chars/value)")
            if stats['unknown_fields']:
                names = ', '.join(sorted(stats['unknown_fields'])[:8])
                print(f"  NOTE: {len(stats['unknown_fields'])} field(s) appeared only after "
                      f"the schema scan and were NOT indexed: {names}. "
                      f"Re-run 'meta rebuild' if you need them.")
        self._metaindex = metaindex.MetadataIndex(path)
        # Self-heal indexes built before nested-dict flattening: if record 0
        # carries a dict column whose children the index never saw (no dotted
        # field names), the index predates the meta.* format -- rebuild it.
        # Datasets using the legacy bare-name 'metadata' convention are
        # unaffected (their field set didn't change), so e.g. the 13M-record
        # AO3 index never rebuilds over this.
        if not rebuild and not getattr(self, '_metaindex_nested_checked', False):
            self._metaindex_nested_checked = True
            try:
                _gi, rec0 = next(self._iter_records_streaming())
                nested = [k for k, v in rec0.items()
                          if isinstance(v, dict) and v and k != 'metadata']
                if nested and not any('.' in f for f in self._metaindex.fields):
                    print(f"  index predates nested-field flattening "
                          f"({', '.join(sorted(nested))} unindexed) -- rebuilding "
                          f"(full corpus pass)...")
                    return self.get_metaindex(rebuild=True)
            except Exception as e:
                print(f"  (nested-field staleness check failed: {e} -- "
                      f"use 'meta rebuild' / the web rebuild button if "
                      f"nested fields are missing)")
        if self._metaindex.n_rows != (self.metadata.get('num_rows') or
                                      self._metaindex.n_rows):
            print(f"  WARNING: index has {self._metaindex.n_rows:,} rows but the dataset "
                  f"reports {self.metadata['num_rows']:,}. Record numbers may not line up; "
                  f"'meta rebuild' will regenerate it.")
        return self._metaindex

    def find_records_by_metadata(self, query: str, limit: Optional[int] = None) -> List[int]:
        """Evaluate a metadata query and return GLOBAL record indices.

        Returns the same shape as find_all_records, so results feed the existing search
        state and are navigable with list / next / prev / goto with no extra plumbing.
        """
        idx = self.get_metaindex()
        node = metaquery.parse(query, idx.fields)
        mask = node.evaluate(idx)
        hits = np.flatnonzero(np.asarray(mask)).astype(int).tolist()
        return hits[:limit] if limit is not None else hits

    def _neardupe_artifact_path(self) -> Path:
        base = self.original_filepath
        cache_dir = (base if base.is_dir() else base.parent) / '.dataset_explorer_cache'
        cache_dir.mkdir(exist_ok=True)
        tag = hashlib.md5(str(base.absolute()).encode()).hexdigest()[:8]
        stem = base.name if base.is_dir() else base.stem
        return cache_dir / f"{stem}_{tag}.neardupe.gz"

    def _neardupe_sig_paths(self) -> Tuple[Path, Path]:
        """Signature and cardinality files, stored as raw .npy beside the artifact.

        Deliberately NOT inside the gzipped artifact. Signatures are essentially random
        bits, so gzip spends real time achieving ~no compression on gigabytes of them.
        Raw .npy is instant to write, and memory-mappable to read -- which is what lets
        sketching stream to disk and matching reload without a 1.7 GB copy.
        """
        art = self._neardupe_artifact_path()
        stem = art.name[:-len('.neardupe.gz')]
        return (art.parent / f"{stem}.neardupe-sig.npy",
                art.parent / f"{stem}.neardupe-card.npy")

    # Sketch-resume sidecar cadence: durable state at most this stale on a crash.
    SKETCH_PERSIST_INTERVAL_S = 30.0

    def _sketch_for_dedup(self, ngram: int, perms: int, field: Optional[str],
                          sample: Optional[int], device: str, exact_cardinality: bool,
                          resume_ok: bool = True):
        total = self.metadata.get('num_rows') or 0
        if sample:
            total = min(total, sample) if total else sample

        use_gpu = device != 'cpu' and neardupe.TORCH_AVAILABLE
        if use_gpu:
            import torch
            use_gpu = torch.cuda.is_available()
        backend = f"GPU ({device})" if use_gpu else "CPU (numpy)"
        src = "raw token shards" if self._npy_sources() else "text field"
        print(f"Sketching {total:,} docs from {src} on {backend} "
              f"[{ngram}-gram, k={perms}]...")
        # Name the feed path and the batch geometry up front: when a sketch is slow,
        # the answer is almost always one of these two lines.
        if not self._npy_sources():
            if self.tok_kind and self.tok_path:
                print(f"  text -> ids via tokenizer.encode ({self.tok_kind}) -- NOTE: "
                      f"much slower than the default word-hash feed. Unless you need "
                      f"shard-identical shingles, reopen without --tok-* for text dedup.")
            else:
                print("  text -> ids via whitespace word-hash")
        if use_gpu:
            import torch
            _dev = torch.device(device)
            free_b, total_b = torch.cuda.mem_get_info(_dev)
            bt = neardupe._auto_batch_tokens(_dev)
            note = ""
            if free_b < 0.25 * total_b:
                note = ("  <- LOW free VRAM shrinks batches; is a training run "
                        "sharing this GPU?")
            print(f"  VRAM {free_b / 1e9:.1f} GB free / {total_b / 1e9:.1f} GB total"
                  f" -> batch {bt:,} tokens{note}")

        fn = neardupe.sketch_corpus_gpu if use_gpu else neardupe.sketch_corpus
        kwargs = dict(n=ngram, k=perms)
        if use_gpu:
            kwargs.update(device=device, exact_cardinality=exact_cardinality)

        # Stream signatures straight into memory-mapped .npy files. Two problems solved at
        # once: nothing accumulates in the interpreter (no per-document objects, no GC
        # cliff, no final stack that doubles peak memory), and the signatures are ALREADY
        # durable on disk when sketching ends -- so a later failure cannot discard hours
        # of work. Requires a known document count, which metadata gives us.
        #
        # SKETCH RESUME: a sidecar records how many documents are durably sketched
        # (memmaps are flushed BEFORE it updates, so it never over-claims). On restart
        # with the same parameters, sketching reopens the files and continues instead
        # of re-deriving hours of work; the feed skips finished docs without
        # tokenizing them.
        start = 0
        sidecar = None
        sk_params = None
        if total:
            sig_path, card_path = self._neardupe_sig_paths()
            sig_path.parent.mkdir(parents=True, exist_ok=True)
            sidecar = sig_path.with_name(
                sig_path.name[:-len('-sig.npy')] + '-sketchprog.json')
            sk_params = json.dumps(
                {'ngram': ngram, 'perms': perms, 'field': field, 'sample': sample,
                 'n_docs': int(total)}, sort_keys=True, default=str)
            if resume_ok and sidecar.exists() and sig_path.exists() and card_path.exists():
                try:
                    blob = json.loads(sidecar.read_text(encoding='utf-8'))
                    if blob.get('params') == sk_params and 0 < int(blob.get('docs_done', 0)) < total:
                        out_try = np.lib.format.open_memmap(sig_path, mode='r+')
                        cards_try = np.lib.format.open_memmap(card_path, mode='r+')
                        if (out_try.shape == (total, perms // 8)
                                and cards_try.shape == (total,)):
                            start = int(blob['docs_done'])
                            kwargs['out'], kwargs['out_cards'] = out_try, cards_try
                except Exception as e:
                    print(f"  (sketch resume unavailable: {e})")
                    start = 0
            if start:
                print(f"  resuming sketch at {start:,}/{total:,} "
                      f"({start / total * 100:.1f}%) from the previous run")
            else:
                if sidecar.exists():
                    sidecar.unlink()
                kwargs['out'] = np.lib.format.open_memmap(
                    sig_path, mode='w+', dtype=np.uint8, shape=(total, perms // 8))
                kwargs['out_cards'] = np.lib.format.open_memmap(
                    card_path, mode='w+', dtype=np.int64, shape=(total,))
            print(f"  streaming signatures to {sig_path.name} "
                  f"({total * (perms // 8) / 1e9:.2f} GB)")

        docs = self._iter_dedup_docs(field=field, sample=sample, skip=start)
        kwargs['start'] = start

        last_persist = [time.time()]

        def _persist(done):
            """Make progress durable: data first, then the claim about it."""
            if sidecar is None:
                return
            now = time.time()
            if now - last_persist[0] < self.SKETCH_PERSIST_INTERVAL_S and done < total:
                return
            last_persist[0] = now
            kwargs['out'].flush()
            kwargs['out_cards'].flush()
            tmp = sidecar.with_name(sidecar.name + '.part')
            tmp.write_text(json.dumps({'params': sk_params, 'docs_done': int(done)}),
                           encoding='utf-8')
            tmp.replace(sidecar)

        if RICH_AVAILABLE and self.console and total:
            with Progress(SpinnerColumn(),
                          TextColumn("[progress.description]{task.description}"),
                          BarColumn(),
                          TextColumn("{task.completed:,}/{task.total:,}"),
                          TimeRemainingColumn(), console=self.console) as progress:
                task = progress.add_task("Sketching...", total=total)

                def _rich_cb(i):
                    done = min(i, total)
                    progress.update(task, completed=done)
                    _persist(done)

                sigs, cards = fn(docs, progress=_rich_cb, **kwargs)
        else:
            # The non-Rich path used to drop the progress callback entirely, so a
            # multi-hour sketch reported NOTHING between "Sketching..." and done.
            # Now: structured progress for embedders + a printed line every 30s.
            t0 = time.time()
            last_print = [t0]
            last_state = [start, t0]             # (count, time) at previous print

            def _sketch_progress(i):
                done = min(i, total) if total else i
                report_progress('sketch', done, total)
                _persist(done)
                now = time.time()
                if total and now - last_print[0] >= 30:
                    last_print[0] = now
                    frac = done / total
                    # Rates over THIS run's work only; `done` includes any
                    # resumed prefix, which would otherwise flatter the average.
                    avg = (done - start) / max(now - t0, 1e-9)
                    eta = (total - done) / max(avg, 1e-9)
                    inst = (done - last_state[0]) / max(now - last_state[1], 1e-9)
                    last_state[0], last_state[1] = done, now
                    # Wall-time split: everything not spent inside a batch flush is the
                    # single-threaded Python doc feed (json + text->ids). A high feed
                    # share means the GPU is idle waiting for documents.
                    split = ""
                    flush_s = getattr(fn, 'live_stats', {}).get('flush_s')
                    if flush_s is not None:
                        feed_pct = max(0.0, 1 - flush_s / max(now - t0, 1e-9)) * 100
                        split = f"  [feed {feed_pct:.0f}% / batch {100 - feed_pct:.0f}% of wall]"
                    print(f"  sketched {done:,}/{total:,} ({frac * 100:.1f}%)  "
                          f"{inst:,.0f} docs/s now, {avg:,.0f} avg  "
                          f"eta {eta / 60:.0f} min{split}")

            sigs, cards = fn(docs, progress=_sketch_progress, **kwargs)

        for arr in (kwargs.get('out'), kwargs.get('out_cards')):
            if arr is not None:
                arr.flush()
        if sidecar is not None and sidecar.exists():
            sidecar.unlink()             # complete: resume state is now obsolete
        return sigs, cards

    def find_near_duplicates(self, threshold: float = 0.8, ngram: int = 13,
                             perms: int = 1024, field: Optional[str] = None,
                             sample: Optional[int] = None, device: str = 'cuda',
                             exact_cardinality: bool = False, rebuild: bool = False,
                             save: bool = True, min_tokens: int = 0,
                             set_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find and rank near-duplicate documents. Returns clusters, ranked by size.

        Signatures are persisted in the artifact, so re-running at a DIFFERENT threshold
        reuses them and skips the sketch entirely -- the expensive stage is the corpus
        read, and threshold is a property of the match, not of the signature.
        """
        if not NEARDUPE_AVAILABLE:
            raise RuntimeError("neardupe.py could not be imported; it must sit beside "
                               "dataset_explorer.py")

        artifact = self._neardupe_artifact_path()
        sig_path, card_path = self._neardupe_sig_paths()
        params = {'ngram': ngram, 'perms': perms, 'field': field, 'sample': sample}
        sigs = cards = None

        if artifact.exists() and not rebuild:
            try:
                with gzip.open(artifact, 'rb') as f:
                    cached = pickle.load(f)
                if all(cached.get(key) == val for key, val in params.items()):
                    if sig_path.exists() and cached.get('n_docs'):
                        n = int(cached['n_docs'])
                        sigs = np.load(sig_path, mmap_mode='r')[:n]
                        cards = (np.load(card_path, mmap_mode='r')[:n]
                                 if card_path.exists() else None)
                    elif cached.get('signatures') is not None:
                        # Artifact from before signatures were split out to .npy.
                        sigs = cached['signatures']
                        cards = cached.get('cardinalities')
                    if sigs is not None:
                        msg = (f"Reusing cached signatures for {sigs.shape[0]:,} docs "
                               f"(sketch skipped; threshold is applied at match time)")
                        if self.console and RICH_AVAILABLE:
                            self.console.print(f"[green]{msg}[/green]")
                        else:
                            print(msg)
            except Exception as e:
                print(f"Could not read near-dupe artifact ({e}); re-sketching.")

        if sigs is None:
            # Sketching is document-feed-bound (single-threaded json + text->ids),
            # so it always uses ONE device; 'cuda:all' applies to matching only.
            sigs, cards = self._sketch_for_dedup(
                ngram, perms, field, sample,
                neardupe.parse_devices(device)[0], exact_cardinality,
                resume_ok=not rebuild)
            # CHECKPOINT before matching. Sketching is the expensive, reusable half; the
            # signature files are already written, so recording the parameters that
            # produced them here means a failure in matching costs the match, not the
            # sketch. Previously the artifact was only written after clustering succeeded,
            # so any error past this point discarded everything.
            if save:
                self._save_neardupe_meta(artifact, dict(
                    params, n_docs=int(sigs.shape[0]), threshold=None, clusters=None))
                print(f"  checkpoint: {sigs.shape[0]:,} signatures saved "
                      f"({sig_path.name}); matching can be re-run without re-sketching")

        n_docs = int(sigs.shape[0])
        if n_docs < 2:
            print("Need at least 2 documents to compare.")
            return []

        # Advisory only -- there is no document-count limit. Exact all-pairs always runs;
        # the warning exists so a multi-hour job is not a surprise.
        strategy = neardupe.recommend_strategy(n_docs, perms)
        est = strategy['est_seconds']
        if n_docs > neardupe.BRUTEFORCE_ADVISORY:
            warn = (f"{n_docs:,} docs: exact all-pairs is "
                    f"~{strategy['pair_macs']/1e12:.0f} TMAC, roughly {est/3600:.1f} h on "
                    f"one modern GPU (cost grows with the SQUARE of document count). "
                    f"Running anyway -- results stay exact. Use --sample to scope it.")
            if self.console and RICH_AVAILABLE:
                self.console.print(f"[yellow]{warn}[/yellow]")
            else:
                print(warn)

        eta = f"{est:.0f}s" if est < 90 else f"{est/60:.1f} min" if est < 5400 else f"{est/3600:.1f} h"
        print(f"Matching {n_docs:,} docs at threshold {threshold} "
              f"({strategy['pair_macs']/1e12:.1f} TMAC, ~{eta} on one modern GPU)...")
        if device != 'cpu' and neardupe.TORCH_AVAILABLE:
            import torch
            if torch.cuda.is_available():
                # Free the sketch phase's cached allocations so tile sizing sees real
                # headroom, and say what the card(s) look like -- a low number here is
                # the early warning for a mis-sized (and therefore glacial) match.
                for _d in neardupe.parse_devices(device):
                    _dev = torch.device(_d)
                    with torch.cuda.device(_dev):
                        torch.cuda.empty_cache()
                    free_b, total_b = torch.cuda.mem_get_info(_dev)
                    print(f"  VRAM {free_b / 1e9:.1f} GB free / {total_b / 1e9:.1f} GB "
                          f"total on {_d} before matching")

        # Pairs are folded into clusters as they are produced rather than accumulated.
        # On a heavily duplicated corpus the pair count is unbounded, and holding it --
        # especially on the GPU -- is what makes a run appear to hang.
        band_cards = np.asarray(cards) if cards is not None else None
        match_devs = neardupe.parse_devices(device)
        # Multi-GPU matching creates per-device clusterers internally and merges
        # them into this one, which therefore stays host-side.
        clusterer = neardupe.StreamingClusterer(
            n_docs, cards=band_cards,
            device=None if (device == 'cpu' or len(match_devs) > 1) else match_devs[0])
        # min_tokens is expressed in TOKENS; the estimator works in shingles, and a
        # document of T tokens yields T - ngram + 1 of them.
        min_card = max(0, min_tokens - ngram + 1) if min_tokens else 0
        if min_card and band_cards is not None:
            n_short = int(np.count_nonzero(band_cards < min_card))
            print(f"  excluding {n_short:,} docs under {min_tokens} tokens "
                  f"({n_short / max(n_docs,1) * 100:.1f}%) from matching")

        # Matching-phase checkpoint: a killed server (restart, crash) resumes
        # from the last completed row blocks instead of redoing hours of tiles.
        # Deleted on success; ignored (and overwritten) on any parameter change.
        match_ckpt = None
        ckpt_path = artifact.parent / (
            artifact.name[:-len('.neardupe.gz')] + '.matchckpt.npz')
        if rebuild and ckpt_path.exists():
            ckpt_path.unlink()          # fresh signatures orphan old match state
        if len(match_devs) == 1:
            match_ckpt = neardupe.MatchCheckpoint(
                ckpt_path,
                params={'threshold': threshold, 'ngram': ngram, 'perms': perms,
                        'field': field, 'sample': sample, 'min_card': min_card,
                        'band_slack': 0.9},
                n_docs=n_docs)
            if match_ckpt.try_load():
                clusterer.seed(match_ckpt.baseline)
        else:
            print("  (match checkpointing not yet available with multiple GPUs)")

        if RICH_AVAILABLE and self.console:
            with Progress(SpinnerColumn(),
                          TextColumn("[progress.description]{task.description}"),
                          BarColumn(),
                          TextColumn("{task.completed:,}/{task.total:,} tiles"),
                          TextColumn("pairs: {task.fields[pairs]:,}"),
                          TimeRemainingColumn(), console=self.console) as progress:
                task = progress.add_task("Matching...", total=1, pairs=0)

                def report(done, total, pairs):
                    progress.update(task, completed=done, total=total, pairs=pairs)

                neardupe.all_pairs_bruteforce(
                    sigs, perms, threshold, device=device, progress=report,
                    cards=band_cards, pair_sink=clusterer, min_card=min_card,
                    checkpoint=match_ckpt)
        else:
            def report(done, total, pairs):
                report_progress('match tiles', done, total,
                                note=f"{pairs:,} pairs")
                if done % 200 == 0 or done == total:
                    print(f"  tiles {done:,}/{total:,}  pairs {pairs:,}", end='\r', flush=True)
            # min_card was historically MISSING here, so the non-Rich path (the
            # web server) printed the exclusion line but never applied it.
            neardupe.all_pairs_bruteforce(
                sigs, perms, threshold, device=device, progress=report,
                cards=band_cards, pair_sink=clusterer, min_card=min_card,
                checkpoint=match_ckpt)
            print()
        if match_ckpt is not None:
            match_ckpt.delete()         # completed: resume state is now obsolete

        stats = getattr(neardupe.all_pairs_bruteforce, 'last_run_stats', {})
        if stats:
            print(f"  {stats['tiles']:,} tiles compared "
                  f"({stats['tiles_skipped_pct']:.1f}% of the triangle skipped by length "
                  f"banding, tile={stats['tile_size']:,})")
        print(f"  {clusterer.total_pairs:,} pairs at or above {threshold}")
        clusters = clusterer.clusters()

        if band_cards is not None:
            # Documents too short for the 1-bit estimator to be reliable. Counted per
            # cluster so the summary can discount them without a separate audit pass.
            short = set(np.flatnonzero(
                band_cards < neardupe.MIN_SHINGLES_FOR_1BIT).tolist())
            for c in clusters:
                c['n_short'] = sum(1 for m in c['members'] if m in short)

        self.dupe_state = {
            'clusters': clusters, 'n_docs': n_docs, 'threshold': threshold,
            'ngram': ngram, 'perms': perms, 'field': field, 'sample': sample,
            'n_pairs': int(clusterer.total_pairs),
        }
        # Near-duplicate results become NAMED SETS like any other search, so they can be
        # switched back to, combined, and -- most usefully -- fed to export as an exclude
        # list. Two sets are stored because they answer different questions:
        #   <name>       every document involved in duplication (for inspection)
        #   <name>_cut   the removable subset only (what you would actually drop)
        # The _cut set applies the same safety rules as prune: chains and short-document
        # clusters are skipped, and protected splits are never included.
        members = sorted({m for c in clusters for m in c['members']})
        if members:
            base = set_name or self._auto_dupe_set_name()
            try:
                kill, _dec, _skipped = neardupe.select_kill_set(
                    clusters, cards=band_cards, protected=self._protected_docs(),
                    skip_chains=True, skip_short=True, keep='longest')
            except Exception:
                kill = set()
            self.store_result_set(sorted(kill), f'near-dupe @ {threshold} (removable)',
                                  'neardupe', name=f'{base}_cut', activate=False)
            self.store_result_set(members, f'near-dupe @ {threshold} (cluster members)',
                                  'neardupe', name=base)
            print(f"  sets: {base!r} ({len(members):,} docs in clusters), "
                  f"{base + '_cut'!r} ({len(kill):,} removable)")

        if save:
            self._save_neardupe_meta(artifact, dict(
                params, n_docs=n_docs, threshold=threshold, clusters=clusters))

        return clusters

    def _save_neardupe_meta(self, artifact: Path, payload: Dict[str, Any]):
        """Write the small artifact: parameters plus clusters. Signatures live separately
        as .npy, so this stays tiny and is cheap to rewrite when only the threshold changes."""
        try:
            with gzip.open(artifact, 'wb', compresslevel=6) as f:
                pickle.dump(dict(payload, version=2, created_at=time.time(),
                                 source=str(self.original_filepath.absolute())),
                            f, protocol=pickle.HIGHEST_PROTOCOL)
            size_kb = artifact.stat().st_size / 1024
            if payload.get('clusters') is not None:
                print(f"Saved near-dupe artifact: {artifact.name} ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"Could not save near-dupe artifact: {e}")

    @staticmethod
    def _annotate_containment(clusters, pi, pj, ps, cards):
        """Attach max containment per cluster.

        |A^B| = J*(|A|+|B|)/(1+J), so containment = |A^B|/min(|A|,|B|) follows from the
        Jaccard estimate plus the two cardinalities. Reported because it catches
        excerpt-inside-a-longer-document pairs that Jaccard alone ranks near zero -- but
        MinHash error is ABSOLUTE, so at high size skew (a short excerpt in a long book,
        true J ~ 0.05) the estimate is unreliable. Treat it as a hint, not a score.
        """
        best: Dict[int, float] = {}
        for a, b, j in zip(pi.tolist(), pj.tolist(), ps.tolist()):
            ca, cb = int(cards[a]), int(cards[b])
            if ca <= 0 or cb <= 0 or j <= 0:
                continue
            inter = j * (ca + cb) / (1.0 + j)
            cont = min(1.0, inter / min(ca, cb))
            for d in (a, b):
                if cont > best.get(d, 0.0):
                    best[d] = cont
        for c in clusters:
            c['max_containment'] = max((best.get(m, 0.0) for m in c['members']), default=0.0)

    _SPLIT_RE = re.compile(r'_(train|val|test)_\d+\.npy$', re.I)

    def _doc_splits(self) -> Tuple[List[Optional[str]], Dict[str, int]]:
        """Split name per source file, plus a doc count per split.

        pre_tokenize.py names shards {label}_{split}_{NNNNNN}.npy and writes train and val
        into the SAME directory, so a directory-mode dedup indexes across both. Knowing
        which split each document came from is what lets val be protected.
        """
        srcs = self.source_files if self.is_directory else [self.original_filepath]
        splits: List[Optional[str]] = []
        counts: Dict[str, int] = {}
        for i, p in enumerate(srcs):
            m = self._SPLIT_RE.search(p.name)
            sp = m.group(1).lower() if m else None
            splits.append(sp)
            n = (self.file_record_counts[i] if self.is_directory
                 else (self.metadata.get('num_rows') or 0))
            counts[sp or 'unsplit'] = counts.get(sp or 'unsplit', 0) + n
        return splits, counts

    def _protected_docs(self, protect_splits=('val', 'test')) -> frozenset:
        """Global doc indices belonging to protected splits (never pruned)."""
        if not self.is_directory:
            m = self._SPLIT_RE.search(self.original_filepath.name)
            if m and m.group(1).lower() in protect_splits:
                return frozenset(range(self.metadata.get('num_rows') or 0))
            return frozenset()
        splits, _ = self._doc_splits()
        out: set = set()
        for i, sp in enumerate(splits):
            if sp in protect_splits:
                out.update(range(self.cum_record_counts[i],
                                 self.cum_record_counts[i] + self.file_record_counts[i]))
        return frozenset(out)

    def prune_near_duplicates(self, out_dir: str, write: bool = False,
                              keep: str = 'longest', include_chains: bool = False,
                              include_short: bool = False, protect_val: bool = True,
                              min_shard: int = 5_000_000) -> Dict[str, Any]:
        """Write a deduplicated copy of the dataset to `out_dir`. Dry-run unless write=True.

        The source tree is never modified, so every cut is reversible by re-running with
        different rules. Output shards mirror input shards 1:1, which preserves the shard
        COUNT -- and therefore DocShardWriter's coprime adjustment (gcd(shards, coprime)==1,
        which stops dataloader orbit-sharing). Repacking would silently undo that.
        """
        if not self.dupe_state or not self.dupe_state.get('clusters'):
            raise ValueError("No near-duplicate results. Run 'neardupe' first.")
        if self.file_type == 'parquet':
            raise ValueError("Pruning supports .npy shards and JSONL; parquet not yet.")

        clusters = self.dupe_state['clusters']
        artifact = self._neardupe_artifact_path()
        _sig_path, card_path = self._neardupe_sig_paths()
        cards = None
        if card_path.exists():
            cards = np.load(card_path, mmap_mode='r')[:self.dupe_state['n_docs']]
        elif artifact.exists():
            try:                                  # pre-v2 artifact with cards inlined
                with gzip.open(artifact, 'rb') as f:
                    inline = pickle.load(f).get('cardinalities')
                cards = np.asarray(inline) if inline is not None else None
            except Exception:
                cards = None

        protected = self._protected_docs() if protect_val else frozenset()
        kill, decisions, skipped = neardupe.select_kill_set(
            clusters, cards=cards, protected=protected,
            skip_chains=not include_chains, skip_short=not include_short, keep=keep)

        total = self.dupe_state['n_docs']
        contaminated = sum(1 for d in decisions if d['reason'] == 'contamination')
        _splits, split_counts = self._doc_splits()

        print(f"\nPrune plan for {self.original_filepath}")
        print(f"  documents          {total:,}")
        if len(split_counts) > 1 or 'unsplit' not in split_counts:
            print(f"  splits             " +
                  ", ".join(f"{k}={v:,}" for k, v in sorted(split_counts.items())))
        if protected:
            print(f"  protected (val)    {len(protected):,}  never removed")
        print(f"  to remove          {len(kill):,}  ({len(kill) / max(total, 1) * 100:.2f}%)")
        if contaminated:
            print(f"    of which         {contaminated:,} are TRAIN/VAL CONTAMINATION "
                  f"(train copies of a val document)")
        print(f"  clusters skipped   chain={skipped['chain']:,} short={skipped['short']:,} "
              f"all-protected={skipped['all_protected']:,}")
        if not write:
            print("  DRY RUN -- nothing written. Re-run with --nd-write to commit.")

        result = {'kill': kill, 'decisions': decisions, 'skipped': skipped,
                  'contaminated': contaminated, 'written': False, 'shards': []}
        if not write or not kill:
            if write and not kill:
                print("  Nothing to remove.")
            return result

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if out.resolve() == (self.original_filepath if self.original_filepath.is_dir()
                             else self.original_filepath.parent).resolve():
            raise ValueError("Output directory must differ from the source directory.")

        if self.file_type == 'npy':
            result['shards'] = self._prune_npy(out, kill, min_shard)
        else:
            result['shards'] = self._prune_jsonl(out, kill)

        cut_log = out / 'cut-manifest.jsonl'
        with open(cut_log, 'w') as f:
            for d in decisions:
                f.write(json.dumps(d) + '\n')
        print(f"  Wrote {cut_log.name} ({len(decisions):,} removals recorded)")
        result['written'] = True
        return result

    def _prune_npy(self, out: Path, kill: set, min_shard: int) -> List[Dict[str, Any]]:
        """Rewrite each .npy shard without the killed documents, 1:1 with the source."""
        bos = self._get_npy_tokenizer().bos_id
        srcs = self.source_files if self.is_directory else [self.original_filepath]
        entries: Dict[str, List[Dict[str, Any]]] = {}
        rows: List[Dict[str, Any]] = []
        undersized: List[str] = []

        for i, src in enumerate(srcs):
            report_progress('prune shards', i, len(srcs), main=True)
            base = self.cum_record_counts[i] if self.is_directory else 0
            n_here = (self.file_record_counts[i] if self.is_directory
                      else (self.metadata.get('num_rows') or 0))
            local_kill = {g - base for g in kill if base <= g < base + n_here}

            starts, ends = npy_doc_spans(src, bos)
            arr = np.load(src, mmap_mode='r')
            keep_spans = [(s, e) for j, (s, e) in enumerate(zip(starts, ends))
                          if j not in local_kill]
            new_arr = (np.concatenate([np.asarray(arr[s:e]) for s, e in keep_spans])
                       if keep_spans else np.empty(0, dtype=arr.dtype))

            # Save through an open handle: np.save(path, ...) APPENDS '.npy' when the name
            # does not already end in it, so writing to '<name>.npy.tmp' would silently
            # produce '<name>.npy.tmp.npy' and the rename would find nothing. Writing to a
            # temp then renaming keeps the output tree crash-safe (same as DocShardWriter).
            dest = out / src.name
            tmp = dest.with_name(dest.name + '.tmp')
            with open(tmp, 'wb') as fh:
                np.save(fh, new_arr)
            tmp.replace(dest)

            m = self._SPLIT_RE.search(src.name)
            split = m.group(1).lower() if m else 'train'
            entries.setdefault(split, []).append({
                'file': src.name, 'tokens': int(len(new_arr)), 'docs': len(keep_spans),
                'blake2b16': hashlib.blake2b(new_arr.tobytes(), digest_size=16).hexdigest(),
            })
            rows.append({'file': src.name, 'split': split, 'docs_before': n_here,
                         'docs_after': len(keep_spans), 'tokens_before': int(len(arr)),
                         'tokens_after': int(len(new_arr))})
            if 0 < len(new_arr) < min_shard:
                undersized.append(f"{src.name} ({len(new_arr):,} tokens)")
            print(f"  {src.name}: {n_here:,} -> {len(keep_spans):,} docs, "
                  f"{len(arr):,} -> {len(new_arr):,} tokens")

        # manifest_{split}.json matches DocShardWriter's schema so rechunk_doc_aligned.py
        # --verify-only still works on the pruned tree.
        for split, ents in entries.items():
            man = {'shard_count': len(ents), 'tokens': sum(e['tokens'] for e in ents),
                   'docs': sum(e['docs'] for e in ents),
                   'dtype': str(np.load(srcs[0], mmap_mode='r').dtype),
                   'generator': 'dataset_explorer.prune_near_duplicates',
                   'doc_order': 'source order, duplicates removed',
                   'shards': ents}
            with open(out / f'manifest_{split}.json', 'w') as f:
                json.dump(man, f, indent=1)
            print(f"  Wrote manifest_{split}.json ({len(ents)} shards, "
                  f"{man['tokens']:,} tokens, {man['docs']:,} docs)")

        if undersized:
            msg = (f"  WARNING: {len(undersized)} shard(s) fell below --min-shard "
                   f"({min_shard:,} tokens). The dataloader SKIPS shards under B*T+1, so "
                   f"these may be silently ignored: " + ", ".join(undersized[:5]))
            if self.console and RICH_AVAILABLE:
                self.console.print(f"[yellow]{msg}[/yellow]")
            else:
                print(msg)
        print("  NOTE: token_counts.json intentionally not written -- the dataloader "
              "fingerprints shard size/mtime and rescans automatically.")
        return rows

    def _prune_jsonl(self, out: Path, kill: set) -> List[Dict[str, Any]]:
        """Copy JSONL sources, dropping killed records by global index."""
        srcs = self.working_files if self.is_directory else [self.filepath]
        names = self.source_files if self.is_directory else [self.original_filepath]
        rows = []
        for i, path in enumerate(srcs):
            report_progress('prune files', i, len(srcs), main=True)
            base = self.cum_record_counts[i] if self.is_directory else 0
            dest = out / names[i].name
            kept = dropped = 0
            with open(path, 'rb') as fin, open(dest, 'wb') as fout:
                for ln, line in enumerate(fin):
                    if (base + ln) in kill:
                        dropped += 1
                        continue
                    fout.write(line)
                    kept += 1
            rows.append({'file': names[i].name, 'docs_after': kept, 'dropped': dropped})
            print(f"  {names[i].name}: kept {kept:,}, dropped {dropped:,}")
        return rows

    def display_dupe_clusters(self, top: int = 25, width: int = 70):
        """Print near-duplicate clusters ranked by match count."""
        if not self.dupe_state or not self.dupe_state.get('clusters'):
            print("No near-duplicate results. Run 'neardupe' first.")
            return
        st = self.dupe_state
        clusters = st['clusters']
        shown = min(top, len(clusters))
        dup_docs = sum(c['size'] for c in clusters)
        removable = sum(c['size'] - 1 for c in clusters)

        # Quality signals, computed here so a run is self-reporting and does not need a
        # separate audit pass before the number can be trusted.
        chains = [c for c in clusters
                  if c['size'] > 2 and c.get('density', 1.0) < neardupe.CHAIN_DENSITY]
        chain_docs = sum(c['size'] for c in chains)
        tainted = [c for c in clusters if c.get('n_short', 0)]
        tainted_docs = sum(c['size'] for c in tainted)
        solid = removable - sum(c['size'] - 1 for c in chains)

        header = (f"{len(clusters):,} clusters covering {dup_docs:,} docs "
                  f"({dup_docs / max(st['n_docs'], 1) * 100:.2f}% of {st['n_docs']:,}); "
                  f"{removable:,} removable if one kept per cluster")
        # Previews work in raw-shard mode too now: one lazy decode per row shown.
        can_preview = True

        if RICH_AVAILABLE and self.console:
            from rich.markup import escape
            table = Table(title=f"Near-duplicate clusters (threshold {st['threshold']})")
            table.add_column("#", style="dim", justify="right")
            table.add_column("Size", style="green", justify="right")
            table.add_column("Max J", style="yellow", justify="right")
            # Density replaces mean similarity: the mean is bounded below by the threshold
            # (only pairs above it are recorded), so it cannot distinguish a tight group
            # from a chain. Density can.
            table.add_column("Dens", style="yellow", justify="right")
            table.add_column("Cont", style="magenta", justify="right")
            table.add_column("Flag", style="red")
            table.add_column("Records", style="cyan")
            if can_preview:
                table.add_column("Preview", style="white")
            for i, c in enumerate(clusters[:shown]):
                recs = ", ".join(str(m) for m in c['members'][:6])
                if c['size'] > 6:
                    recs += f", +{c['size'] - 6}"
                flags = []
                if c['size'] > 2 and c.get('density', 1.0) < neardupe.CHAIN_DENSITY:
                    flags.append("chain")
                if c.get('n_short', 0):
                    flags.append(f"short×{c['n_short']}")
                row = [str(i + 1), f"{c['size']:,}", f"{c['max_sim']:.3f}",
                       f"{c.get('density', 1.0):.2f}",
                       f"{c.get('max_containment', 0.0):.2f}", " ".join(flags), recs]
                if can_preview:
                    row.append(escape(self._record_preview_line(c['members'][0], width)))
                table.add_row(*row)
            self.console.print(table)
            self.console.print(header)
        else:
            print(f"\nNear-duplicate clusters (threshold {st['threshold']}):")
            for i, c in enumerate(clusters[:shown]):
                recs = ", ".join(str(m) for m in c['members'][:6])
                if c['size'] > 6:
                    recs += f", +{c['size'] - 6}"
                flags = []
                if c['size'] > 2 and c.get('density', 1.0) < neardupe.CHAIN_DENSITY:
                    flags.append("chain")
                if c.get('n_short', 0):
                    flags.append(f"short x{c['n_short']}")
                line = (f"  {i+1:>4}. size={c['size']:<5} max={c['max_sim']:.3f} "
                        f"dens={c.get('density', 1.0):.2f} "
                        f"cont={c.get('max_containment', 0.0):.2f} "
                        f"{'[' + ' '.join(flags) + ']' if flags else '':<14} [{recs}]")
                if can_preview:
                    line += f"  {self._record_preview_line(c['members'][0], width)}"
                print(line)
            print(header)

        if shown < len(clusters):
            print(f"... ({len(clusters) - shown:,} more; 'dupes all' or 'dupes <n>')")

        # Quality summary. These are the two ways the headline count misleads, so they are
        # reported on every run rather than left to a separate audit.
        if chains:
            print(f"  CHAINS: {len(chains):,} clusters ({chain_docs:,} docs, "
                  f"{chain_docs / max(dup_docs, 1) * 100:.1f}% of clustered) have edge "
                  f"density < {neardupe.CHAIN_DENSITY} -- these are transitive chains, not "
                  f"tight groups. Their members are not all duplicates of each other.")
        if tainted:
            pct = tainted_docs / max(dup_docs, 1) * 100
            note = ("  WARNING" if pct > 10 else "  NOTE")
            print(f"{note}: {len(tainted):,} clusters ({tainted_docs:,} docs, {pct:.1f}%) "
                  f"contain documents under {neardupe.MIN_SHINGLES_FOR_1BIT} shingles, "
                  f"where the 1-bit estimator is unreliable and densification can invent "
                  f"similarity." + (" Raise --nd-perms or exclude short docs before "
                                    "trusting these." if pct > 10 else ""))
        if not chains and not tainted:
            print("  All clusters are internally tight and long enough to trust.")
        print(f"  BOTTOM LINE: {removable:,} removable; ~{max(solid, 0):,} of those sit in "
              f"internally-tight clusters. Inspect with 'dupe <n>' before cutting.")

    def show_dupe_cluster(self, rank: int, truncate: bool = True):
        """Display every member of cluster `rank` (1-based)."""
        if not self.dupe_state or not self.dupe_state.get('clusters'):
            print("No near-duplicate results. Run 'neardupe' first.")
            return
        clusters = self.dupe_state['clusters']
        if rank < 1 or rank > len(clusters):
            print(f"Cluster {rank} out of range (1-{len(clusters):,})")
            return
        c = clusters[rank - 1]
        print(f"\nCluster {rank}: {c['size']} docs, max J={c['max_sim']:.3f}, "
              f"mean J={c['mean_sim']:.3f}, containment={c.get('max_containment', 0.0):.2f}")
        for m in c['members']:
            try:
                self.display_records(self.get_record(m), truncate=truncate)
            except Exception as e:
                print(f"  record #{m}: {e}")

    def _show_current_match(self, truncate: bool = True):
        """Display the current match from the active search_state."""
        if not self.search_state or not self.search_state.get('indices'):
            print("No active search results. Run 'findall <query>' first.")
            return

        s = self.search_state
        cursor = s['cursor']
        total = len(s['indices'])
        record_idx = s['indices'][cursor]

        header = f">>> Match {cursor + 1:,} of {total:,}  |  record #{record_idx:,}"
        if s.get('field'):
            header += f"  |  field: {s['field']}"
        if s.get('regex'):
            header += "  |  regex"
        header += f"  |  query: {s['query']!r}"

        if RICH_AVAILABLE and self.console:
            self.console.print(f"[bold yellow]{header}[/bold yellow]")
        else:
            print("\n" + header)

        try:
            df = self.get_record(record_idx)
            self.display_records(df, truncate=truncate)
        except Exception as e:
            print(f"Error fetching record #{record_idx}: {e}")

    def _display_mode_str(self) -> str:
        """Human-readable summary of the current display setting."""
        if self.full_display:
            return "FULL"
        return f"TRUNCATED@{self.max_display_width}"

    def _navigate_match(self, delta: int, truncate: bool = True):
        """Move the search cursor by delta and display."""
        if not self.search_state or not self.search_state.get('indices'):
            print("No active search results. Run 'findall <query>' first.")
            return
        total = len(self.search_state['indices'])
        new_cursor = self.search_state['cursor'] + delta
        if new_cursor < 0:
            print(f"Already at first match (1 of {total:,})")
            return
        if new_cursor >= total:
            print(f"Already at last match ({total:,} of {total:,})")
            return
        self.search_state['cursor'] = new_cursor
        self._show_current_match(truncate=truncate)

    def _record_preview_line(self, record_idx: int, width: int) -> str:
        """Return a single-line preview of a record for list display."""
        try:
            df = self.get_record(record_idx)
        except Exception as e:
            return f"<error: {e}>"
        if df.empty:
            return "<empty record>"
        row = df.iloc[0]

        field = (self.search_state or {}).get('field')
        text = None
        if field and field in row.index:
            val = row[field]
            text = "" if val is None else str(val)
        else:
            for candidate in ('text', 'content', 'message', 'body'):
                if candidate in row.index and row[candidate] is not None:
                    text = str(row[candidate])
                    break
            if text is None:
                parts = []
                for col in row.index:
                    val = row[col]
                    if val is None:
                        continue
                    parts.append(f"{col}={val}")
                text = " | ".join(parts)

        # Flatten to a single line with visible escapes so the preview shows the first
        # `width` chars of actual CONTENT (not just the first physical line). Newlines/tabs
        # are rendered as literal \n / \t / \r rather than truncating at the first break.
        flat = text.strip().replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        if len(flat) > width:
            flat = flat[:max(width - 3, 1)] + "..."
        return flat

    def list_matches(self, count: Optional[int] = None, width: int = 100):
        """Print one line per match: '<n>: <preview>'."""
        if not self.search_state or not self.search_state.get('indices'):
            print("No active search results. Run 'findall <query>' first.")
            return
        indices = self.search_state['indices']
        total = len(indices)
        n = total if count is None else min(count, total)
        idx_width = len(f"{total:,}")
        cursor = self.search_state.get('cursor', 0)

        if RICH_AVAILABLE and self.console:
            from rich.markup import escape
            for i in range(n):
                marker = "*" if i == cursor else " "
                preview = self._record_preview_line(indices[i], width)
                self.console.print(
                    f"{marker} [bold]{i + 1:>{idx_width},}[/bold]  "
                    f"[dim]rec #{indices[i]:,}[/dim]  {escape(preview)}"
                )
        else:
            for i in range(n):
                marker = "*" if i == cursor else " "
                preview = self._record_preview_line(indices[i], width)
                print(f"{marker} {i + 1:>{idx_width},}  rec #{indices[i]:,}  {preview}")

        if n < total:
            print(f"... ({total - n:,} more; use 'list all' or 'list <n>' to see more)")
        else:
            print(f"({total:,} match{'es' if total != 1 else ''} total)")

    def find_record_number(self, byte_position: int) -> int:
        """Find the exact record number for a given byte position."""
        if self.is_directory:
            print("findrec is not supported in directory mode (byte positions are per-file).")
            return -1
        if self.file_type != 'jsonl':
            print("Record number lookup only works for JSONL files")
            return -1
        
        # If we have an index, use binary search for O(log n) lookup!
        if self.line_positions:
            print(f"Finding record using index (fast)...")
            
            # Binary search
            left, right = 0, len(self.line_positions) - 1
            
            while left <= right:
                mid = (left + right) // 2
                mid_pos = self.line_positions[mid]
                
                if mid_pos == byte_position:
                    print(f"Found exact match! Byte position {byte_position:,} is record #{mid}")
                    return mid
                elif mid_pos < byte_position:
                    left = mid + 1
                else:
                    right = mid - 1
            
            # Not exact match, find closest
            if right >= 0 and right < len(self.line_positions):
                print(f"Byte position {byte_position:,} is closest to record #{right}")
                return right
            
            print(f"Byte position {byte_position:,} is beyond file")
            return -1
        
        # Fallback to linear search
        print(f"Finding record number for byte position {byte_position:,} (no index)...")
        print("This may take a moment for large files...")
        
        record_num = 0
        bytes_read = 0
        
        progress_interval = 100000
        next_progress = progress_interval
        
        with open(self.filepath, 'rb') as f:
            while bytes_read < byte_position:
                line = f.readline()
                if not line:
                    break
                
                bytes_read += len(line)
                record_num += 1
                
                if record_num >= next_progress:
                    percent = (bytes_read / byte_position) * 100
                    print(f"  Checked {record_num:,} records ({percent:.1f}% to target)...")
                    next_progress += progress_interval
            
            if bytes_read >= byte_position:
                print(f"Found it! Byte position {byte_position:,} is in record #{record_num:,}")
                
                f.seek(byte_position)
                if byte_position > 0:
                    f.readline()
                line = f.readline()
                if line:
                    try:
                        record = json.loads(line.decode('utf-8'))
                        print(f"Record preview: {str(record)[:100]}...")
                    except:
                        pass
                
                return record_num
            else:
                print(f"Byte position {byte_position:,} is beyond end of file")
                return -1
    
    def print_info(self):
        """Print basic file information."""
        if RICH_AVAILABLE:
            title_name = (self.original_filepath.name + "/"
                          if self.is_directory else self.original_filepath.name)
            table = Table(title=f"Dataset Info: {title_name}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("File Type", self.file_type.upper())

            # Show compression / conversion / directory info if applicable
            if self.is_directory:
                table.add_row("Source", "Directory (multi-file)")
                table.add_row("Number of Files", f"{self.metadata['num_files']:,}")
                table.add_row("Total Source Size", f"{self.metadata['total_source_mb']:.2f} MB")
                if abs(self.metadata['total_working_mb'] - self.metadata['total_source_mb']) > 0.01:
                    table.add_row("Total Working Size", f"{self.metadata['total_working_mb']:.2f} MB")
            elif self.is_compressed:
                table.add_row("Original Format", self.original_filepath.suffix.upper())
                table.add_row("Compressed Size", f"{self.metadata['original_file_size']:.2f} MB")
                table.add_row("Decompressed Size", f"{self.metadata['decompressed_file_size']:.2f} MB")
                table.add_row("Compression Ratio", f"{self.metadata['compression_ratio']:.1f}x")
            elif self.is_json_array:
                table.add_row("Original Format", "JSON array (converted to JSONL)")
                table.add_row("Original Size", f"{self.metadata['original_file_size']:.2f} MB")
                table.add_row("Converted Size", f"{self.metadata['converted_file_size']:.2f} MB")
            else:
                table.add_row("File Size", f"{self.metadata['file_size']:.2f} MB")

            if self.metadata.get('num_rows') is not None:
                count_str = f"{self.metadata['num_rows']:,}"
                if self.metadata.get('count_is_estimate'):
                    count_str = f"~{count_str} (estimated)"
                table.add_row("Total Records", count_str)
            else:
                table.add_row("Total Records", "Not counted")

            table.add_row("Number of Fields", str(self.metadata['num_columns']))

            # Add cache/index info
            if self.file_type == 'jsonl':
                if self.metadata.get('has_index'):
                    table.add_row("Index Status", "✓ Built (fast random access)")
                else:
                    table.add_row("Index Status", "✗ Not built")

            if self.is_directory:
                cached = sum(1 for c in self.file_caches if c and c.cache_filepath.exists())
                table.add_row("Cache Status",
                              f"✓ {cached}/{len(self.file_caches)} files cached"
                              if cached else "✗ No per-file caches")
            elif self.cache and self.cache.cache_filepath.exists():
                cache_size = self.cache.cache_filepath.stat().st_size / 1024
                table.add_row("Cache Status", f"✓ Cached ({cache_size:.1f} KB)")
            else:
                table.add_row("Cache Status", "✗ Not cached")

            # Show temp file location for compressed/converted files
            if (self.is_compressed or self.is_json_array) and not self.is_directory:
                table.add_row("Temp File", str(self.filepath))

            self.console.print(table)

            # Per-file breakdown in directory mode
            if self.is_directory:
                files_table = Table(title="Files (alphabetical)")
                files_table.add_column("#", style="dim", justify="right")
                files_table.add_column("Name", style="cyan")
                files_table.add_column("Records", style="green", justify="right")
                files_table.add_column("Size (MB)", style="yellow", justify="right")
                files_table.add_column("Note", style="magenta")
                for i, info in enumerate(self.file_metadata_list):
                    note_parts = []
                    if info.get('is_compressed'):
                        note_parts.append("compressed")
                    if info.get('is_json_array'):
                        note_parts.append("json-array")
                    files_table.add_row(
                        str(i + 1),
                        info['source_path'].name,
                        f"{info['num_rows']:,}",
                        f"{info['source_size_mb']:.2f}",
                        ", ".join(note_parts),
                    )
                self.console.print(files_table)

            schema_table = Table(title="Schema")
            schema_table.add_column("Field", style="cyan")
            schema_table.add_column("Type", style="yellow")

            for field, dtype in self.metadata['schema'].items():
                schema_table.add_row(field, dtype)

            self.console.print(schema_table)
        else:
            print("\n" + "="*50)
            label = (self.original_filepath.name + "/"
                     if self.is_directory else self.original_filepath.name)
            print(f"Dataset Info: {label}")
            print("="*50)
            print(f"File Type: {self.file_type.upper()}")

            if self.is_directory:
                print("Source: Directory (multi-file)")
                print(f"Number of Files: {self.metadata['num_files']:,}")
                print(f"Total Source Size: {self.metadata['total_source_mb']:.2f} MB")
                if abs(self.metadata['total_working_mb'] - self.metadata['total_source_mb']) > 0.01:
                    print(f"Total Working Size: {self.metadata['total_working_mb']:.2f} MB")
            elif self.is_compressed:
                print(f"Original Format: {self.original_filepath.suffix.upper()}")
                print(f"Compressed Size: {self.metadata['original_file_size']:.2f} MB")
                print(f"Decompressed Size: {self.metadata['decompressed_file_size']:.2f} MB")
                print(f"Compression Ratio: {self.metadata['compression_ratio']:.1f}x")
            elif self.is_json_array:
                print(f"Original Format: JSON array (converted to JSONL)")
                print(f"Original Size: {self.metadata['original_file_size']:.2f} MB")
                print(f"Converted Size: {self.metadata['converted_file_size']:.2f} MB")
            else:
                print(f"File Size: {self.metadata['file_size']:.2f} MB")

            if self.metadata.get('num_rows') is not None:
                count_str = f"{self.metadata['num_rows']:,}"
                if self.metadata.get('count_is_estimate'):
                    count_str = f"~{count_str} (estimated)"
                print(f"Total Records: {count_str}")
            else:
                print("Total Records: Not counted")

            print(f"Number of Fields: {self.metadata['num_columns']}")

            if self.file_type == 'jsonl':
                if self.metadata.get('has_index'):
                    print("Index Status: ✓ Built (fast random access)")
                else:
                    print("Index Status: ✗ Not built")

            if self.is_directory:
                cached = sum(1 for c in self.file_caches if c and c.cache_filepath.exists())
                if cached:
                    print(f"Cache Status: ✓ {cached}/{len(self.file_caches)} files cached")
                else:
                    print("Cache Status: ✗ No per-file caches")
            elif self.cache and self.cache.cache_filepath.exists():
                cache_size = self.cache.cache_filepath.stat().st_size / 1024
                print(f"Cache Status: ✓ Cached ({cache_size:.1f} KB)")
            else:
                print("Cache Status: ✗ Not cached")

            if (self.is_compressed or self.is_json_array) and not self.is_directory:
                print(f"Temp File: {self.filepath}")

            if self.is_directory:
                print("\nFiles:")
                for i, info in enumerate(self.file_metadata_list, 1):
                    notes = []
                    if info.get('is_compressed'):
                        notes.append("compressed")
                    if info.get('is_json_array'):
                        notes.append("json-array")
                    note_str = f"  ({', '.join(notes)})" if notes else ""
                    print(f"  {i:>3}. {info['source_path'].name}: "
                          f"{info['num_rows']:,} records, "
                          f"{info['source_size_mb']:.2f} MB{note_str}")

            print("\nSchema:")
            for field, dtype in self.metadata['schema'].items():
                print(f"  - {field}: {dtype}")
    
    def display_records(self, df: pd.DataFrame, truncate: bool = True):
        """Display records in a nice format."""
        # Determine index type from DataFrame attribute if available
        index_type = getattr(df, '_index_type', None)
        
        if RICH_AVAILABLE:
            from rich.markup import escape
            for idx, row in df.iterrows():
                try:
                    panel_content = ""
                    for col, val in row.items():
                        val_str = str(val)
                        if truncate and len(val_str) > self.max_display_width:
                            val_str = val_str[:self.max_display_width] + "..."
                        val_str = escape(val_str)
                        panel_content += f"[cyan]{escape(str(col))}:[/cyan] {val_str}\n"
                    
                    # Determine title based on index type
                    if index_type == 'byte_position':
                        title = f"Random Sample (byte offset: {idx:,})"
                    else:
                        # Default to record number for everything else
                        title = f"Record #{idx}"
                    
                    self.console.print(Panel(panel_content.rstrip(), title=title))
                    
                except Exception as e:
                    # Fallback to simple display if Rich has issues
                    if index_type == 'byte_position':
                        print(f"\n=== Random Sample (byte offset: {idx:,}) === (Rich display failed)")
                    else:
                        print(f"\n=== Record #{idx} === (Rich display failed)")
                    
                    for col, val in row.items():
                        val_str = str(val)
                        if truncate and len(val_str) > self.max_display_width:
                            val_str = val_str[:self.max_display_width] + "..."
                        print(f"{col}: {val_str}")
        else:
            # Non-Rich display
            for idx, row in df.iterrows():
                # Determine title based on index type
                if index_type == 'byte_position':
                    print(f"\n=== Random Sample (byte offset: {idx:,}) ===")
                else:
                    print(f"\n=== Record #{idx} ===")
                
                for col, val in row.items():
                    val_str = str(val)
                    if truncate and len(val_str) > self.max_display_width:
                        val_str = val_str[:self.max_display_width] + "..."
                    print(f"{col}: {val_str}")
    
    def get_statistics(self, field: Optional[str] = None) -> Dict[str, Any]:
        """Get statistical information about the dataset or a specific field."""
        stats = {}
        
        if self.file_type == 'parquet':
            if field:
                if self.is_directory:
                    parts = [pd.read_parquet(p, columns=[field]) for p in self.working_files]
                    df_col = pd.concat(parts, ignore_index=True)
                else:
                    df_col = pd.read_parquet(self.filepath, columns=[field])
                col = df_col[field]
                
                stats['field'] = field
                stats['non_null_count'] = col.notna().sum()
                stats['null_count'] = col.isna().sum()
                stats['null_percentage'] = (col.isna().sum() / len(col)) * 100
                
                if pd.api.types.is_numeric_dtype(col):
                    stats['type'] = 'numeric'
                    stats['mean'] = col.mean()
                    stats['median'] = col.median()
                    stats['std'] = col.std()
                    stats['min'] = col.min()
                    stats['max'] = col.max()
                    stats['quantiles'] = col.quantile([0.25, 0.5, 0.75]).to_dict()
                else:
                    stats['type'] = 'categorical/text'
                    stats['unique_values'] = col.nunique()
                    stats['most_common'] = col.value_counts().head(10).to_dict()
                    
                    if col.dtype == 'object':
                        text_lengths = col.dropna().astype(str).str.len()
                        stats['avg_length'] = text_lengths.mean()
                        stats['min_length'] = text_lengths.min()
                        stats['max_length'] = text_lengths.max()
            else:
                # Sample from first file for whole-dataset stats; mirror legacy behavior.
                source_for_sample = self.working_files[0] if self.is_directory else self.filepath
                df_sample = pd.read_parquet(source_for_sample).head(10000)
                stats['sample_size'] = len(df_sample)
                stats['memory_usage_mb'] = df_sample.memory_usage(deep=True).sum() / (1024 * 1024)
                
                type_counts = df_sample.dtypes.value_counts()
                stats['column_types'] = {str(k): v for k, v in type_counts.items()}
                
                null_counts = df_sample.isnull().sum()
                stats['null_counts'] = null_counts[null_counts > 0].to_dict()
        
        elif self.file_type == 'jsonl':
            field_values = []
            record_sizes = []

            if self.is_directory:
                total = self.metadata.get('num_rows') or 0
                sample_size = min(10000, total)
                for i in range(sample_size):
                    record = self.get_record_by_position(i)
                    if record:
                        # Approximate size from JSON re-encoding (per-file byte index
                        # would require cross-file bookkeeping; close enough for stats).
                        record_sizes.append(len(json.dumps(record)))
                        if field and field in record:
                            field_values.append(record[field])
            elif self.line_positions:
                sample_size = min(10000, len(self.line_positions))
                for i in range(sample_size):
                    record = self.get_record_by_position(i)
                    if record:
                        if i < len(self.line_positions) - 1:
                            record_size = self.line_positions[i+1] - self.line_positions[i]
                        else:
                            record_size = len(json.dumps(record))
                        record_sizes.append(record_size)
                        if field and field in record:
                            field_values.append(record[field])
            else:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 10000:
                            break
                        record = json.loads(line)
                        record_sizes.append(len(line))
                        if field and field in record:
                            field_values.append(record[field])
            
            if field:
                stats['field'] = field
                stats['non_null_count'] = len([v for v in field_values if v is not None])
                stats['null_count'] = len([v for v in field_values if v is None])
                
                non_null_values = [v for v in field_values if v is not None]
                if non_null_values:
                    if all(isinstance(v, (int, float)) for v in non_null_values):
                        stats['type'] = 'numeric'
                        stats['mean'] = np.mean(non_null_values)
                        stats['median'] = np.median(non_null_values)
                        stats['std'] = np.std(non_null_values)
                        stats['min'] = min(non_null_values)
                        stats['max'] = max(non_null_values)
                    else:
                        stats['type'] = 'text/mixed'
                        str_values = [str(v) for v in non_null_values]
                        stats['unique_values'] = len(set(str_values))
                        value_counts = Counter(str_values)
                        stats['most_common'] = dict(value_counts.most_common(10))
                        
                        text_lengths = [len(s) for s in str_values]
                        stats['avg_length'] = np.mean(text_lengths)
                        stats['min_length'] = min(text_lengths)
                        stats['max_length'] = max(text_lengths)
            else:
                stats['avg_record_size_bytes'] = np.mean(record_sizes)
                stats['min_record_size_bytes'] = min(record_sizes)
                stats['max_record_size_bytes'] = max(record_sizes)
        
        return stats
    
    def display_statistics(self, stats: Dict[str, Any]):
        """Display statistics in a nice format."""
        if RICH_AVAILABLE:
            table = Table(title="Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in stats.items():
                if isinstance(value, dict):
                    value_str = "\n".join([f"{k}: {v}" for k, v in value.items()])
                elif isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                
                table.add_row(key.replace('_', ' ').title(), value_str)
            
            self.console.print(table)
        else:
            print("\n" + "="*50)
            print("Statistics")
            print("="*50)
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"{key.replace('_', ' ').title()}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                elif isinstance(value, float):
                    print(f"{key.replace('_', ' ').title()}: {value:.4f}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value}")
    
    def interactive_mode(self):
        """Run interactive exploration mode."""
        print("\n" + "="*60)
        print("Dataset Explorer - Interactive Mode")
        print("="*60)
        if self.is_directory:
            print(f"Loaded directory: {self.original_filepath}")
            print(f"  {self.metadata['num_files']:,} files, "
                  f"{self.metadata['num_rows']:,} records total")
        else:
            print(f"Loaded: {self.original_filepath.name}")

        if self.is_compressed:
            print(f"Format: Compressed {self.original_filepath.suffix.upper()}")
            print(f"Working with decompressed temp file in: {self.filepath.parent}")
        elif self.is_json_array:
            print(f"Format: JSON array (converted to JSONL)")
            print(f"Working with converted temp file in: {self.filepath.parent}")

        if self.is_directory:
            cached = sum(1 for c in self.file_caches if c and c.cache_filepath.exists())
            print(f"Cache: {cached}/{len(self.file_caches)} files cached")
        elif self.cache:
            if self.cache.cache_filepath.exists():
                print(f"Cache: {self.cache.cache_filepath.name}")
            else:
                print("Cache: Not yet created")

        print("Commands: info, sample, record, findall, list, goto, sets, switch, meta, "
              "neardupe, dupes, stats, export, cache, help, quit")
        print("="*60)
        
        while True:
            try:
                raw_command = input("\n> ").strip()
                command = raw_command.lower()

                if command == 'quit' or command == 'exit':
                    print("Goodbye!")
                    break
                
                elif command == 'help':
                    print("\nAvailable commands:")
                    print("  info                       - Show dataset information")
                    print("  sample [n] [random]        - Sample n records (default 5)")
                    print("  sample [n] full            - Sample n records without truncation")
                    print("  record <number>            - Show specific record by number (0-based)")
                    print("  record <number> full       - Show specific record without truncation")
                    print("  findrec <byte_pos>         - Find record number at byte position")
                    print("  findall <query>            - Find ALL matching records (steppable)")
                    print("  findall -f <field> <query> - Find matches in a specific field")
                    print("  findall -r <regex>         - Regex match (combine with -f as needed)")
                    print("  findall -m <query>         - METADATA query (see 'meta fields')")
                    print("      e.g. findall -m ('Category': 'Gen' OR 'Characters')")
                    print("                       AND 'author': SUBSTR('xenak') AND 'words' < 100")
                    print("      :  contains    =  exact match of one comma-separated element")
                    print("      <  >  <=  >=  numeric    ~  regex    EXISTS('field')")
                    print("      AND / OR / NOT, parentheses; always case-insensitive")
                    print("  findall -n <name> ...      - Name the result set (any search kind)")
                    print("  sets                       - List result sets (count + query)")
                    print("  sets clear                 - Delete all result sets")
                    print("  switch <name>              - Make a saved set the active results")
                    print("  rename <old> <new>         - Rename a result set")
                    print("  delete <name>              - Delete one result set")
                    print("  meta fields                - List metadata fields (counts only)")
                    print("  meta values <field>        - Most common values of one field")
                    print("  meta rebuild [<field>]     - Rebuild the index; optionally name")
                    print("                               the body field explicitly")
                    print("                               (raw token shards: findall searches TOKEN")
                    print("                                sequences -- no decode, full coverage, but")
                    print("                                case/whitespace sensitive; -r and -f N/A)")
                    print("  findall -n <N> <query>     - Cap at N matches (fast peek; 0 = unlimited)")
                    print("  findall -a t1 t2 ...       - AND match: every term must be present")
                    print("                               (quote terms with spaces, e.g. -a \"foo bar\" baz)")
                    print("                               live per-term tally shown (hits~ = approx,")
                    print("                               short-circuited); add -c for EXACT counts (slower)")
                    print("  next, n                    - Step to next match")
                    print("  prev, p                    - Step to previous match")
                    print("  goto <n>                   - Jump to match #n (1-based)")
                    print("  list [<n>|all] [<width>]   - List matches, one line each")
                    print("                               (defaults: 200 matches, 100 chars)")
                    print("  full                       - Toggle full (untruncated) display")
                    print("  full on|off                - Set full display explicitly")
                    print("  full <N>                   - Truncate at N chars (medium mode)")
                    print("  compact                    - Use the current truncation width")
                    print("  maxdisplay <N>             - Set truncation width (turns full off)")
                    print("  results                    - Show current match status")
                    print("  neardupe [opts]            - Find & rank near-duplicate documents")
                    print("                               -t <thr>    similarity threshold (0.8)")
                    print("                               -g <ngram>  shingle size in tokens (13)")
                    print("                               -k <perms>  signature bits, pow2 (1024)")
                    print("                               -f <field>  text field (non-tokenized data)")
                    print("                               --sample N  only the first N documents")
                    print("                               --cpu       force CPU sketching")
                    print("                               --exact-card exact distinct-shingle counts")
                    print("                               --rebuild   ignore cached signatures")
                    print("                               -m <N>      skip docs under N tokens")
                    print("  neardupe -n <name> ...     - Name the resulting sets:")
                    print("                               <name> = docs in clusters,")
                    print("                               <name>_cut = removable subset")
                    print("                               (-n is a NAME everywhere; ngram is -g)")
                    print("  dupes [<n>|all]            - List clusters, ranked by match count")
                    print("  prune <out_dir> [opts]     - Write a deduplicated COPY to out_dir")
                    print("                               (dry-run unless --write; source untouched)")
                    print("                               --keep longest|first  survivor rule")
                    print("                               --include-chains  also cut low-density clusters")
                    print("                               --include-short   also cut short-doc clusters")
                    print("                               --prune-val       allow removing val docs")
                    print("  dupe <n> [full]            - Show every member of cluster n")
                    print("  stats [field]              - Show statistics")
                    print("  export <n> <file>          - Export first n records to file")
                    print("  export record <n> <f>      - Export specific record to file")
                    print("  cache clear                - Clear cached metadata")
                    print("  cache rebuild              - Rebuild cache with full index")
                    print("  cache info                 - Show cache information")
                    print("  quit                       - Exit the program")
                
                elif command == 'info':
                    self.print_info()
                
                elif command.startswith('cache'):
                    parts = command.split()
                    if len(parts) < 2:
                        print("Usage: cache [clear|rebuild|info]")
                        continue

                    # Directory mode: per-file caches
                    if self.is_directory:
                        active_caches = [c for c in self.file_caches if c is not None]
                        if not active_caches:
                            print("Cache is disabled")
                            continue
                        if parts[1] == 'clear':
                            for c in active_caches:
                                c.clear()
                            print(f"Cleared cache for {len(active_caches)} files.")
                        elif parts[1] == 'rebuild':
                            print("Rebuilding caches by re-running file preparation...")
                            # Reset multi-file state and reinitialize
                            self.source_files.clear()
                            self.working_files.clear()
                            self.file_record_counts.clear()
                            self.file_line_positions.clear()
                            self.file_metadata_list.clear()
                            self.file_caches.clear()
                            self.cum_record_counts = [0]
                            self.rebuild_cache = True
                            try:
                                self._init_multi_file()
                            finally:
                                self.rebuild_cache = False
                        elif parts[1] == 'info':
                            cached_files = [c for c in active_caches if c.cache_filepath.exists()]
                            total_kb = sum(c.cache_filepath.stat().st_size for c in cached_files) / 1024
                            print(f"Per-file caches: {len(cached_files)}/{len(active_caches)} present")
                            print(f"Total cache size: {total_kb:.1f} KB")
                        else:
                            print("Usage: cache [clear|rebuild|info]")
                        continue

                    if parts[1] == 'clear':
                        if self.cache:
                            self.cache.clear()
                        else:
                            print("Cache is disabled")

                    elif parts[1] == 'rebuild':
                        if self.cache:
                            print("Rebuilding cache with full index...")
                            self.cache.clear()

                            if self.file_type == 'jsonl':
                                line_count, self.line_positions = self._build_line_positions_with_progress(self.filepath)
                                self.metadata['num_rows'] = line_count
                                self.metadata['count_is_estimate'] = False
                                self.metadata['has_index'] = True
                                print(f"Total records: {line_count:,}")

                                self.cache.save(self.metadata, self.line_positions)
                            else:
                                self._load_metadata()
                        else:
                            print("Cache is disabled")

                    elif parts[1] == 'info':
                        if self.cache:
                            if self.cache.cache_filepath.exists():
                                cache_size = self.cache.cache_filepath.stat().st_size
                                print(f"Cache file: {self.cache.cache_filepath}")
                                print(f"Cache size: {cache_size / 1024:.1f} KB")

                                cache_data = self.cache.load()
                                if cache_data:
                                    cached_time = cache_data.get('cached_at', 0)
                                    if cached_time:
                                        age_hours = (time.time() - cached_time) / 3600
                                        print(f"Cache age: {age_hours:.1f} hours")

                                    if cache_data.get('line_positions'):
                                        print(f"Index entries: {len(cache_data['line_positions']):,}")
                            else:
                                print("No cache exists")
                        else:
                            print("Cache is disabled")
                
                elif command.startswith('record'):
                    parts = command.split()
                    if len(parts) < 2:
                        print("Usage: record <number> [full]")
                        continue
                    
                    try:
                        index = int(parts[1])
                        truncate = 'full' not in parts
                        
                        df = self.get_record(index)
                        print(f"\nShowing record #{index}:")
                        self.display_records(df, truncate=truncate)
                    
                    except ValueError as e:
                        print(f"Error: {e}")
                    except Exception as e:
                        print(f"Error fetching record: {e}")
                
                elif command.startswith('maxdisplay'):
                    parts = command.split()
                    if len(parts) < 2:
                        print("Usage: maxdisplay <width>")
                        continue

                    try:
                        width = int(parts[1])
                    except ValueError:
                        print(f"Invalid width: {parts[1]}")
                        continue
                    if width <= 0:
                        print("Width must be a positive integer.")
                        continue
                    self.max_display_width = width
                    self.full_display = False
                    print(f"Display: {self._display_mode_str()}")
                    if self.search_state and self.search_state.get('indices'):
                        self._show_current_match(truncate=not self.full_display)
                
                elif command.startswith('findrec') or command.startswith('findrecord'):
                    parts = command.split()
                    if len(parts) < 2:
                        print("Usage: findrec <byte_position>")
                        print("Example: findrec 99277231312")
                        continue
                    
                    try:
                        byte_pos_str = parts[1].replace(',', '')
                        byte_pos = int(byte_pos_str)
                        
                        record_num = self.find_record_number(byte_pos)
                        
                        if record_num >= 0:
                            show = input(f"\nDisplay record #{record_num}? (y/n): ").strip().lower()
                            if show == 'y':
                                df = self.get_record(record_num)
                                self.display_records(df, truncate=True)
                    
                    except ValueError:
                        print(f"Invalid byte position: {parts[1]}")
                    except Exception as e:
                        print(f"Error finding record: {e}")
                
                elif command.startswith('sample'):
                    parts = command.split()
                    n = 5
                    random = False
                    truncate = True
                    
                    if len(parts) > 1:
                        try:
                            n = int(parts[1])
                        except ValueError:
                            pass
                    
                    if 'random' in parts:
                        random = True
                    
                    if 'full' in parts:
                        truncate = False
                    
                    df = self.sample_records(n, random)
                    if random:
                        print(f"\nShowing {len(df)} random samples{' (full text)' if not truncate else ''}:")
                    else:
                        print(f"\nShowing {len(df)} records{' (full text)' if not truncate else ''}:")
                    self.display_records(df, truncate=truncate)
                
                elif _META_FINDALL_RE.match(raw_command.strip()):
                    _mm = _META_FINDALL_RE.match(raw_command.strip())
                    set_name = _mm.group('name')
                    expr = raw_command.strip()[_mm.end():].strip()
                    if not METAQUERY_AVAILABLE:
                        print("metaquery.py / metaindex.py must sit beside dataset_explorer.py")
                        continue
                    try:
                        t0 = time.time()
                        idx = self.get_metaindex()
                        node = metaquery.parse(expr, idx.fields)
                        hits = self.find_records_by_metadata(expr)
                        dt = time.time() - t0
                    except metaquery.QueryError as e:
                        print(f"Query error: {e}")
                        continue
                    except Exception as e:
                        print(f"Metadata search failed: {e}")
                        continue
                    if not hits:
                        print(f"No records matched. ({dt:.2f}s)")
                        if set_name:
                            # An explicitly named query is worth recording even when empty:
                            # "this category is absent" is a curation finding, and the set
                            # documents the query that established it.
                            self.store_result_set([], expr, 'metadata', name=set_name)
                            print(f"  stored empty set {set_name!r}")
                        else:
                            self.search_state = None
                        continue
                    stored = self.store_result_set(hits, expr, 'metadata', name=set_name)
                    self.search_state['meta_fields'] = sorted(node.fields())
                    pct = len(hits) / max(idx.n_rows, 1) * 100
                    print(f"Found {len(hits):,} of {idx.n_rows:,} records ({pct:.2f}%) "
                          f"in {dt:.2f}s -> set {stored!r}. Showing match 1 "
                          f"(display: {self._display_mode_str()}).")
                    print("Navigate: 'next'/'n', 'prev'/'p', 'goto <n>', 'list', "
                          "'full' to toggle.")
                    self._show_current_match(truncate=not self.full_display)

                elif command == 'sets' or command.startswith('sets '):
                    parts = raw_command.split()
                    if len(parts) > 1 and parts[1].lower() == 'clear':
                        n = len(self.result_sets)
                        self.result_sets.clear()
                        self._save_sets()
                        self.search_state = None
                        print(f"Cleared {n} result set{'s' if n != 1 else ''}.")
                    else:
                        self.display_result_sets()

                elif command.startswith('switch'):
                    parts = raw_command.split()
                    if len(parts) < 2:
                        print("Usage: switch <set name>   ('sets' to list)")
                        continue
                    if self.activate_result_set(parts[1]):
                        self._show_current_match(truncate=not self.full_display)

                elif command.startswith('delete') or command.startswith('drop '):
                    parts = raw_command.split()
                    if len(parts) < 2:
                        print("Usage: delete <set name>")
                        continue
                    self.delete_result_set(parts[1])

                elif command.startswith('rename'):
                    parts = raw_command.split()
                    if len(parts) < 3:
                        print("Usage: rename <old> <new>")
                        continue
                    old_n, new_n = parts[1], parts[2]
                    if old_n not in self.result_sets:
                        print(f"No result set named {old_n!r}.")
                        continue
                    if new_n in self.result_sets:
                        print(f"A set named {new_n!r} already exists.")
                        continue
                    self.result_sets[new_n] = self.result_sets.pop(old_n)
                    if (self.search_state or {}).get('set_name') == old_n:
                        self.search_state['set_name'] = new_n
                    self._save_sets()
                    print(f"Renamed {old_n!r} -> {new_n!r}.")

                elif command == 'meta' or command.startswith('meta '):
                    if not METAQUERY_AVAILABLE:
                        print("metaquery.py / metaindex.py must sit beside dataset_explorer.py")
                        continue
                    parts = raw_command.split()
                    sub = parts[1].lower() if len(parts) > 1 else 'fields'
                    try:
                        if sub == 'rebuild':
                            if len(parts) > 2:
                                self.text_field = raw_command.split(None, 2)[2].strip().strip("'\"")
                                print(f"  body field set to {self.text_field!r}")
                            self._metaindex = None
                            self.get_metaindex(rebuild=True)
                            continue
                        idx = self.get_metaindex()
                        if sub == 'fields':
                            # Immediate feedback: this used to run silently for minutes on a
                            # large index, which is indistinguishable from a hang.
                            print(f"Summarising {len(idx.fields)} fields over "
                                  f"{idx.n_rows:,} records...")
                            rows = idx.field_summary()
                            if RICH_AVAILABLE and self.console:
                                tbl = Table(title=f"Metadata fields ({idx.n_rows:,} records)")
                                tbl.add_column("Field", style="cyan")
                                tbl.add_column("Non-empty", style="green", justify="right")
                                tbl.add_column("Fill %", style="green", justify="right")
                                tbl.add_column("Distinct", style="yellow", justify="right")
                                tbl.add_column("Origin", style="magenta")
                                for r in rows:
                                    tbl.add_row(r['field'], f"{r['non_empty']:,}",
                                                f"{r['fill_pct']:.1f}", f"{r['distinct']:,}",
                                                "computed" if r['derived'] else "source")
                                self.console.print(tbl)
                            else:
                                for r in rows:
                                    print(f"  {r['field']:<24} {r['non_empty']:>10,}  "
                                          f"{r['fill_pct']:>6.1f}%  {r['distinct']:,} distinct"
                                          f"{'  [computed]' if r['derived'] else ''}")
                            derived = [r['field'] for r in rows if r['derived']]
                            if derived:
                                print()
                                print("Computed fields (added at index time, not present "
                                      "in the source metadata):")
                                for f in derived:
                                    print(f"  {f:<16} {metaindex.DERIVED_FIELD_DOC[f]}")
                                print("  e.g.  'words' < 100        size filter covering "
                                      "every record, not just those declaring a count")
                                print("        'words_ratio' < 0.5  records whose text "
                                      "failed to extract")
                                print()
                            if rows and not rows[0]['exact']:
                                print(f"Estimated from {rows[0]['sampled']:,} sampled rows "
                                      f"of {idx.n_rows:,} (row groups spread across the "
                                      f"file); 'distinct' is the count within the sample.")
                            print("Counts only. Use 'meta values <field>' to see values "
                                  "for one field you name.")
                        elif sub == 'values':
                            if len(parts) < 3:
                                print("Usage: meta values <field> [top]")
                                continue
                            field = raw_command.split(None, 2)[2].strip().strip("'\"")
                            top = 20
                            if ' ' in field and field.rsplit(' ', 1)[1].isdigit():
                                field, top = field.rsplit(' ', 1)[0], int(field.rsplit(' ', 1)[1])
                            vc = idx.value_counts(field.strip().strip("'\""), top=top)
                            for val, cnt in vc.items():
                                shown = (str(val)[:80] or '<empty>')
                                print(f"  {cnt:>10,}  {shown}")
                        else:
                            print("Usage: meta [fields|values <field>|rebuild]")
                    except KeyError as e:
                        print(f"No key named {e}. Try 'meta fields'.")
                    except Exception as e:
                        print(f"Metadata command failed: {e}")

                elif command.startswith('findall'):
                    args_str = raw_command[len('findall'):].strip()
                    try:
                        tokens = shlex.split(args_str, posix=True)
                    except ValueError as e:
                        print(f"Could not parse arguments (unbalanced quotes?): {e}")
                        continue
                    field: Optional[str] = None
                    result_set_name: Optional[str] = None
                    use_regex = False
                    set_full = False
                    match_all = False
                    count_terms = False
                    limit: Optional[int] = None
                    i = 0
                    parse_error = None
                    while i < len(tokens):
                        tok = tokens[i]
                        if tok in ('-f', '--field'):
                            if i + 1 >= len(tokens):
                                parse_error = "Missing field name after -f"
                                break
                            field = tokens[i + 1]
                            i += 2
                        elif tok in ('-n', '--name'):
                            if i + 1 >= len(tokens):
                                parse_error = "Missing set name after -n"
                                break
                            result_set_name = tokens[i + 1]
                            i += 2
                        elif tok in ('-r', '--regex'):
                            use_regex = True
                            i += 1
                        elif tok in ('-a', '--all'):
                            match_all = True
                            i += 1
                        elif tok in ('-c', '--count-terms'):
                            count_terms = True
                            i += 1
                        elif tok in ('-n', '--limit'):
                            if i + 1 >= len(tokens):
                                parse_error = "Missing number after -n"
                                break
                            try:
                                limit = int(tokens[i + 1])
                            except ValueError:
                                parse_error = f"Invalid limit: {tokens[i + 1]}"
                                break
                            if limit <= 0:
                                limit = None  # 0 or negative means unlimited
                            i += 2
                        elif tok == 'full':
                            set_full = True
                            i += 1
                        else:
                            break
                    if parse_error:
                        print(parse_error)
                        continue

                    rest = tokens[i:]
                    if match_all:
                        terms = rest
                    else:
                        # Single-term mode: rejoin so spaces in the query are preserved
                        # (mainly meaningful when the user did NOT quote — quoted terms
                        # are already a single token).
                        if len(rest) == 1:
                            terms = rest
                        elif len(rest) > 1:
                            terms = [' '.join(rest)]
                        else:
                            terms = []

                    if not terms:
                        prompt_query = input("Enter search query: ").strip()
                        if not prompt_query:
                            print("Empty query, aborted.")
                            continue
                        terms = [prompt_query]

                    if field is not None:
                        resolved = self._resolve_field(field)
                        if resolved is None:
                            print(f"Field '{field}' not found. Available fields:")
                            print(', '.join(self.metadata.get('columns') or []))
                            continue
                        field = resolved

                    if len(terms) > 1:
                        desc = f"Finding records matching ALL of {terms!r}"
                    else:
                        desc = f"Finding records matching {terms[0]!r}"
                    if field:
                        desc += f" in field '{field}'"
                    if use_regex:
                        desc += " (regex)"
                    if limit is not None:
                        desc += f" (limit {limit:,})"
                    print(desc + " ...")

                    try:
                        query_arg = terms if len(terms) > 1 else terms[0]
                        match_indices = self.find_all_records(
                            query_arg, field=field, regex=use_regex, limit=limit,
                            count_terms=count_terms
                        )
                    except ValueError as e:
                        print(f"Error: {e}")
                        continue

                    if not match_indices:
                        print("No matches found.")
                        self.search_state = None
                        continue

                    if set_full:
                        self.full_display = True

                    limit_hit = limit is not None and len(match_indices) >= limit
                    qtext = str(terms if len(terms) > 1 else terms[0])
                    kind = 'token' if self.file_type == 'npy' else 'text'
                    stored = self.store_result_set(match_indices, qtext, kind,
                                                   name=result_set_name)
                    self.search_state['field'] = field
                    self.search_state['regex'] = use_regex
                    self.search_state['limit_hit'] = limit_hit
                    suffix = " (--limit hit, more may exist)" if limit_hit else ""
                    print(f"Found {len(match_indices):,} matches{suffix} -> set {stored!r}. "
                          f"Showing match 1 (display: {self._display_mode_str()}).")
                    print("Navigate: 'next'/'n', 'prev'/'p', 'goto <n>', 'list', 'results', 'full' to toggle.")
                    self._show_current_match(truncate=not self.full_display)

                elif command == 'next' or command == 'n':
                    self._navigate_match(+1, truncate=not self.full_display)

                elif command == 'prev' or command == 'p':
                    self._navigate_match(-1, truncate=not self.full_display)

                elif command.startswith('goto'):
                    parts = command.split()
                    if len(parts) < 2:
                        print("Usage: goto <match_number>")
                        continue
                    if not self.search_state or not self.search_state.get('indices'):
                        print("No active search results. Run 'findall <query>' first.")
                        continue
                    try:
                        n = int(parts[1])
                    except ValueError:
                        print(f"Invalid match number: {parts[1]}")
                        continue
                    total = len(self.search_state['indices'])
                    if n < 1 or n > total:
                        print(f"Match number {n} out of range (1-{total:,})")
                        continue
                    self.search_state['cursor'] = n - 1
                    self._show_current_match(truncate=not self.full_display)

                elif command == 'full' or command.startswith('full '):
                    parts = command.split()
                    arg = parts[1] if len(parts) > 1 else None
                    if arg in ('on', 'true', '1', 'yes'):
                        self.full_display = True
                    elif arg in ('off', 'false', '0', 'no'):
                        self.full_display = False
                    elif arg is None:
                        self.full_display = not self.full_display
                    else:
                        # Try numeric width: 'full 1000' => truncate at 1000 chars
                        try:
                            width = int(arg)
                        except ValueError:
                            print("Usage: full [on|off|<width-in-chars>]   (no arg toggles)")
                            continue
                        if width <= 0:
                            print("Width must be a positive integer.")
                            continue
                        self.max_display_width = width
                        self.full_display = False
                    print(f"Display: {self._display_mode_str()}")
                    if self.search_state and self.search_state.get('indices'):
                        self._show_current_match(truncate=not self.full_display)

                elif command == 'compact':
                    self.full_display = False
                    print(f"Display: {self._display_mode_str()}")
                    if self.search_state and self.search_state.get('indices'):
                        self._show_current_match(truncate=not self.full_display)

                elif command == 'results':
                    if not self.search_state or not self.search_state.get('indices'):
                        print("No active search results. Run 'findall <query>' first.")
                        continue
                    s = self.search_state
                    total = len(s['indices'])
                    line = f"Match {s['cursor'] + 1:,} of {total:,}"
                    if s.get('limit_hit'):
                        line += " (--limit hit)"
                    line += f"  |  query: {s['query']!r}"
                    if s.get('field'):
                        line += f"  |  field: {s['field']}"
                    if s.get('regex'):
                        line += "  |  regex"
                    line += f"  |  current record #: {s['indices'][s['cursor']]:,}"
                    line += f"  |  display: {self._display_mode_str()}"
                    print(line)

                elif command == 'list' or command.startswith('list '):
                    parsed = _parse_list_args(command.split()[1:])
                    if parsed is not None:
                        count, width = parsed
                        self.list_matches(count=count, width=width)

                elif command == 'neardupe' or command.startswith('neardupe '):
                    if not NEARDUPE_AVAILABLE:
                        print("neardupe.py not found beside dataset_explorer.py")
                        continue
                    try:
                        tokens = shlex.split(raw_command[len('neardupe'):].strip(), posix=True)
                    except ValueError as e:
                        print(f"Could not parse arguments: {e}")
                        continue
                    opts = {'threshold': 0.8, 'ngram': 13, 'perms': 1024, 'field': None,
                            'sample': None, 'device': 'cuda', 'exact_cardinality': False,
                            'rebuild': False, 'min_tokens': 0}
                    nd_set_name = None
                    # -n means NAME everywhere in this tool, so ngram is -g. A bare number
                    # after -n is almost certainly someone reaching for the old spelling.
                    numeric = {'-t': ('threshold', float), '--threshold': ('threshold', float),
                               '-g': ('ngram', int), '--ngram': ('ngram', int),
                               '-k': ('perms', int), '--perms': ('perms', int),
                               '--sample': ('sample', int),
                               '-m': ('min_tokens', int), '--min-tokens': ('min_tokens', int)}
                    i, bad = 0, None
                    while i < len(tokens):
                        tok = tokens[i]
                        if tok in numeric:
                            if i + 1 >= len(tokens):
                                bad = f"Missing value after {tok}"
                                break
                            key, cast = numeric[tok]
                            try:
                                opts[key] = cast(tokens[i + 1])
                            except ValueError:
                                bad = f"Invalid value for {tok}: {tokens[i + 1]}"
                                break
                            i += 2
                        elif tok in ('-f', '--field'):
                            if i + 1 >= len(tokens):
                                bad = "Missing field name after -f"
                                break
                            opts['field'] = tokens[i + 1]
                            i += 2
                        elif tok in ('-n', '--name'):
                            if i + 1 >= len(tokens):
                                bad = "Missing set name after -n"
                                break
                            if tokens[i + 1].isdigit():
                                bad = (f"-n takes a result-set NAME, but got "
                                       f"{tokens[i + 1]!r}. Shingle size is -g.")
                                break
                            nd_set_name = tokens[i + 1]
                            i += 2
                        elif tok == '--cpu':
                            opts['device'] = 'cpu'; i += 1
                        elif tok == '--exact-card':
                            opts['exact_cardinality'] = True; i += 1
                        elif tok == '--rebuild':
                            opts['rebuild'] = True; i += 1
                        else:
                            bad = f"Unknown option: {tok}"
                            break
                    if bad:
                        print(bad)
                        print("Usage: neardupe [-n <set name>] [-t <thr>] [-g <ngram>] "
                              "[-k <perms>] [-m <min tokens>] [-f <field>] [--sample N] "
                              "[--cpu] [--exact-card] [--rebuild]")
                        continue
                    if not 0 < opts['threshold'] <= 1:
                        print("Threshold must be in (0, 1].")
                        continue
                    if opts['perms'] & (opts['perms'] - 1):
                        print(f"perms must be a power of two (got {opts['perms']}).")
                        continue
                    try:
                        t0 = time.time()
                        clusters = self.find_near_duplicates(**opts, set_name=nd_set_name)
                        print(f"Done in {time.time() - t0:.1f}s")
                        if clusters:
                            self.display_dupe_clusters()
                            print("Navigate members with 'list', 'next', 'goto'; "
                                  "inspect a cluster with 'dupe <n>'.")
                        else:
                            print("No near-duplicates found at this threshold.")
                    except Exception as e:
                        print(f"Near-duplicate detection failed: {e}")

                elif command == 'dupes' or command.startswith('dupes '):
                    parts = command.split()
                    top = 25
                    if len(parts) > 1:
                        if parts[1] == 'all':
                            top = 10 ** 9
                        else:
                            try:
                                top = int(parts[1])
                            except ValueError:
                                print(f"Invalid count: {parts[1]}")
                                continue
                    self.display_dupe_clusters(top=top)

                elif command.startswith('prune'):
                    try:
                        tokens = shlex.split(raw_command[len('prune'):].strip(), posix=True)
                    except ValueError as e:
                        print(f"Could not parse arguments: {e}")
                        continue
                    opts = {'write': False, 'keep': 'longest', 'include_chains': False,
                            'include_short': False, 'protect_val': True}
                    out_dir, bad = None, None
                    i = 0
                    while i < len(tokens):
                        tok = tokens[i]
                        if tok == '--write':
                            opts['write'] = True; i += 1
                        elif tok == '--include-chains':
                            opts['include_chains'] = True; i += 1
                        elif tok == '--include-short':
                            opts['include_short'] = True; i += 1
                        elif tok == '--prune-val':
                            opts['protect_val'] = False; i += 1
                        elif tok == '--keep':
                            if i + 1 >= len(tokens) or tokens[i + 1] not in ('longest', 'first'):
                                bad = "--keep takes 'longest' or 'first'"
                                break
                            opts['keep'] = tokens[i + 1]; i += 2
                        elif tok.startswith('-'):
                            bad = f"Unknown option: {tok}"
                            break
                        else:
                            out_dir = tok; i += 1
                    if bad or not out_dir:
                        print(bad or "Usage: prune <out_dir> [--write] [--keep longest|first]")
                        print("       [--include-chains] [--include-short] [--prune-val]")
                        continue
                    try:
                        self.prune_near_duplicates(out_dir, **opts)
                    except Exception as e:
                        print(f"Prune failed: {e}")

                elif command.startswith('dupe '):
                    parts = command.split()
                    try:
                        rank = int(parts[1])
                    except (IndexError, ValueError):
                        print("Usage: dupe <cluster_number> [full]")
                        continue
                    self.show_dupe_cluster(rank, truncate=('full' not in parts))

                elif command.startswith('stats'):
                    parts = command.split()
                    field = None
                    
                    if len(parts) > 1:
                        field = parts[1]
                        if field not in self.metadata['columns']:
                            print(f"Field '{field}' not found. Available fields:")
                            print(", ".join(self.metadata['columns']))
                            continue
                    
                    print(f"Computing statistics{f' for {field}' if field else ''}...")
                    stats = self.get_statistics(field)
                    self.display_statistics(stats)
                
                elif command.startswith('export'):
                    parts = command.split()
                    
                    if len(parts) > 1 and parts[1] == 'record':
                        if len(parts) < 4:
                            print("Usage: export record <number> <filename>")
                            continue
                        
                        try:
                            index = int(parts[2])
                            filename = parts[3]
                            
                            df = self.get_record(index)
                            
                            if filename.endswith('.csv'):
                                df.to_csv(filename, index=False)
                            elif filename.endswith('.json'):
                                df.to_json(filename, orient='records', indent=2)
                            elif filename.endswith('.parquet'):
                                df.to_parquet(filename, index=False)
                            else:
                                filename += '.json'
                                df.to_json(filename, orient='records', indent=2)
                            
                            print(f"Exported record #{index} to {filename}")
                        
                        except ValueError as e:
                            print(f"Error: {e}")
                        except Exception as e:
                            print(f"Export failed: {e}")
                    
                    else:
                        if len(parts) < 3:
                            print("Usage: export <n> <filename>")
                            print("   or: export record <number> <filename>")
                            continue
                        
                        try:
                            n = int(parts[1])
                            filename = parts[2]
                            
                            df = self.sample_records(n)
                            
                            if filename.endswith('.csv'):
                                df.to_csv(filename, index=False)
                            elif filename.endswith('.json'):
                                df.to_json(filename, orient='records', indent=2)
                            elif filename.endswith('.parquet'):
                                df.to_parquet(filename, index=False)
                            else:
                                filename += '.csv'
                                df.to_csv(filename, index=False)
                            
                            print(f"Exported {len(df)} records to {filename}")
                        
                        except ValueError:
                            print("Invalid number of records")
                        except Exception as e:
                            print(f"Export failed: {e}")
                
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")
            
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Dataset Explorer - A flashlight for large dataset files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
        Examples:
            # Interactive mode (with caching)
            python dataset_explorer.py data.parquet
            python dataset_explorer.py huge_data.jsonl.zst
            
            # Quick mode (estimates for large JSONL files)
            python dataset_explorer.py huge_data.jsonl --quick
            
            # Rebuild cache with full index
            python dataset_explorer.py data.jsonl --rebuild-cache
            
            # Disable cache
            python dataset_explorer.py data.jsonl --no-cache
            
            # Show info and exit
            python dataset_explorer.py data.jsonl.zst --info
            
            # Sample 10 random records (uses cached index if available)
            python dataset_explorer.py data.parquet --sample 10 --random
        
        Supported formats:
            - Parquet files (.parquet)
            - JSONL files (.jsonl, .json)
            - Compressed JSONL files (.jsonl.zst) - auto-decompressed to local tmp/
        
        Note: .zst files are decompressed, JSON arrays converted, and .npy shards decoded
              into a 'tmp' directory beside the source file (cross-platform: Windows/Unix).
              These are mtime-validated caches and are KEPT between runs, so re-opening the
              same source is an instant cache hit. Pass --clear-temp to delete them on exit.
              Install zstandard: pip install zstandard
        ''')
    )
    
    parser.add_argument('file', help='Path to a data file (.parquet, .jsonl, .json, .jsonl.zst) '
                        'OR a directory of same-format files (alphabetical, top-level only)')
    parser.add_argument('--info', action='store_true', help='Show file info and exit')
    parser.add_argument('--migrate-cache', action='store_true',
                        help='Adopt caches/temps/sets/near-dupe artifacts after the '
                             'dataset was moved or copied to this path, then exit. '
                             'Renames path-hash-keyed cache files to the new location.')
    parser.add_argument('--quick', action='store_true', help='Quick mode - estimate counts for large files')
    parser.add_argument('--no-cache', action='store_true', help='Disable metadata caching')
    parser.add_argument('--rebuild-cache', action='store_true', help='Rebuild cache with full index')
    parser.add_argument('--sample', type=int, metavar='N', help='Sample N records')
    parser.add_argument('--record', type=int, metavar='NUMBER', help='Get specific record by number')
    parser.add_argument('--random', action='store_true', help='Random sampling')
    parser.add_argument('--full', action='store_true', help='Show full records without truncation')
    parser.add_argument('--search', type=str, help='Find records matching query (use --field to scope, --limit to cap)')
    parser.add_argument('--field', type=str,
                        help='The text/body field. Scopes --search and --stats, and tells '
                             'the metadata indexer which field is the document body (it is '
                             'excluded from the index and used for word counts). Without '
                             'it the body is detected by value length.')
    parser.add_argument('--stats', nargs='?', const='', help='Show statistics (optionally for specific field)')
    parser.add_argument('--export', type=str, help='Export results to file')
    parser.add_argument('--limit', type=int, default=10,
                        help='Max matches to return for --search (default: 10; 0 = unlimited full scan)')
    # .npy tokenized-shard support: decode back to text via the SAME tokenizer that produced them
    parser.add_argument('--tok-kind', type=str, help='.npy: tokenizer kind (e.g. llama, hf, tiktoken)')
    parser.add_argument('--tok-path', type=str, help='.npy: tokenizer path (dir or model file)')
    parser.add_argument('--special-tokens', type=str, help='.npy: special-tokens config json (optional)')
    parser.add_argument('--npy-max-docs', type=int, help='.npy: decode only the first N docs (quick peek)')
    parser.add_argument('--clear-temp', action='store_true',
                        help='Delete decoded/decompressed temp files in tmp/ on exit. They are '
                             'KEPT by default: they are mtime-validated caches, and re-decoding '
                             'a large shard set is the slowest thing this tool does.')
    parser.add_argument('--keep-temp', action='store_true',
                        help=argparse.SUPPRESS)   # deprecated: keeping temp files is now default
    # Near-duplicate detection (batch form; the interactive command is 'neardupe')
    parser.add_argument('--raw-shards', action='store_true',
                        help='Read .npy token shards RAW instead of decoding them to JSONL. '
                             'Decoding a large shard tree takes hours to days; raw mode opens '
                             'instantly and decodes single records on demand. findall searches '
                             'token sequences instead of text (no -r/-f). Implied by --neardupe.')
    parser.add_argument('--neardupe', action='store_true',
                        help='Find and rank near-duplicate documents, then exit')
    parser.add_argument('--nd-threshold', type=float, default=0.8, metavar='T',
                        help='Near-dupe similarity threshold (default: 0.8)')
    parser.add_argument('--nd-ngram', type=int, default=13, metavar='N',
                        help='Near-dupe shingle size in tokens (default: 13)')
    parser.add_argument('--nd-perms', type=int, default=1024, metavar='K',
                        help='Near-dupe signature bits, power of two (default: 1024)')
    parser.add_argument('--nd-sample', type=int, metavar='N',
                        help='Near-dupe: only the first N documents')
    parser.add_argument('--nd-top', type=int, default=25, metavar='N',
                        help='Near-dupe: clusters to display (default: 25)')
    parser.add_argument('--nd-cpu', action='store_true',
                        help='Near-dupe: force CPU sketching')
    parser.add_argument('--nd-exact-card', action='store_true',
                        help='Near-dupe: exact distinct-shingle counts (slower; for containment)')
    parser.add_argument('--nd-rebuild', action='store_true',
                        help='Near-dupe: ignore cached signatures and re-sketch')
    parser.add_argument('--nd-min-tokens', type=int, default=0, metavar='N',
                        help='Exclude documents shorter than N tokens from matching. Below '
                             'roughly 200 tokens the estimator is unreliable, and a large '
                             'short tail inflates the pair count enough to stall a run.')
    parser.add_argument('--nd-prune', type=str, metavar='OUT_DIR',
                        help='After --neardupe, write a deduplicated COPY to OUT_DIR '
                             '(dry-run unless --nd-write; the source is never modified)')
    parser.add_argument('--nd-write', action='store_true',
                        help='Commit the prune (without this, --nd-prune only reports)')
    parser.add_argument('--nd-keep', choices=['longest', 'first'], default='longest',
                        help='Which cluster member survives a prune (default: longest)')
    parser.add_argument('--nd-include-chains', action='store_true',
                        help='Prune low-density (transitive chain) clusters too -- unsafe')
    parser.add_argument('--nd-include-short', action='store_true',
                        help='Prune clusters containing short documents too -- unsafe')
    parser.add_argument('--nd-prune-val', action='store_true',
                        help='Allow removing val/test documents (default: they are protected)')

    args = parser.parse_args()

    if args.keep_temp:
        print("Note: --keep-temp is deprecated; keeping temp files is now the default. "
              "Use --clear-temp to delete them on exit.")

    if args.migrate_cache:
        try:
            migrate_cache(args.file)
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    try:
        # Check for zstandard if needed
        if args.file.endswith('.zst') and not ZSTD_AVAILABLE:
            print("Error: zstandard library not installed.")
            print("Please install it to work with .zst files:")
            print("  pip install zstandard")
            sys.exit(1)
        
        explorer = DatasetExplorer(
            args.file,
            quick_mode=args.quick,
            no_cache=args.no_cache,
            rebuild_cache=args.rebuild_cache,
            tok_kind=args.tok_kind,
            tok_path=args.tok_path,
            special_tokens=args.special_tokens,
            npy_max_docs=args.npy_max_docs,
            clear_temp=args.clear_temp,
            # Reading .npy shards raw skips the decode-to-JSONL step entirely, which is
            # the dominant cost of opening a large tokenized dataset.
            dedup_only=args.neardupe or args.raw_shards,
            text_field=args.field,
        )

        if args.neardupe:
            if not NEARDUPE_AVAILABLE:
                print("Error: neardupe.py must sit beside dataset_explorer.py")
                sys.exit(1)
            if args.nd_perms & (args.nd_perms - 1):
                print(f"Error: --nd-perms must be a power of two (got {args.nd_perms})")
                sys.exit(1)
            t0 = time.time()
            clusters = explorer.find_near_duplicates(
                threshold=args.nd_threshold, ngram=args.nd_ngram, perms=args.nd_perms,
                field=args.field, sample=args.nd_sample,
                device='cpu' if args.nd_cpu else 'cuda',
                exact_cardinality=args.nd_exact_card, rebuild=args.nd_rebuild,
                min_tokens=args.nd_min_tokens,
            )
            print(f"Done in {time.time() - t0:.1f}s")
            if clusters:
                explorer.display_dupe_clusters(top=args.nd_top)
                if args.export:
                    rows = [{'cluster': i + 1, 'size': c['size'], 'max_jaccard': c['max_sim'],
                             'mean_jaccard': c['mean_sim'],
                             'max_containment': c.get('max_containment', 0.0),
                             'records': ' '.join(str(m) for m in c['members'])}
                            for i, c in enumerate(clusters)]
                    pd.DataFrame(rows).to_csv(args.export, index=False)
                    print(f"Exported {len(rows):,} clusters to {args.export}")
                if args.nd_prune:
                    explorer.prune_near_duplicates(
                        args.nd_prune, write=args.nd_write, keep=args.nd_keep,
                        include_chains=args.nd_include_chains,
                        include_short=args.nd_include_short,
                        protect_val=not args.nd_prune_val)
            else:
                print("No near-duplicates found at this threshold.")

        elif args.info:
            explorer.print_info()
        
        elif args.record is not None:
            df = explorer.get_record(args.record)
            print(f"\nShowing record #{args.record}:")
            explorer.display_records(df, truncate=not args.full)
            
            if args.export:
                if args.export.endswith('.json'):
                    df.to_json(args.export, orient='records', indent=2)
                else:
                    df.to_csv(args.export, index=False)
                print(f"Exported to {args.export}")
        
        elif args.sample:
            df = explorer.sample_records(args.sample, args.random)
            explorer.display_records(df, truncate=not args.full)
            
            if args.export:
                df.to_csv(args.export, index=False)
                print(f"\nExported to {args.export}")
        
        elif args.search:
            field = args.field
            if field is not None:
                resolved = explorer._resolve_field(field)
                if resolved is None:
                    print(f"Field '{field}' not found. Available: "
                          f"{', '.join(explorer.metadata.get('columns') or [])}")
                    sys.exit(1)
                field = resolved

            limit = args.limit if args.limit and args.limit > 0 else None
            indices = explorer.find_all_records(args.search, field=field, limit=limit)

            if not indices:
                print("No matches found.")
            else:
                limit_hit = limit is not None and len(indices) >= limit
                suffix = " (--limit hit, more may exist)" if limit_hit else ""
                print(f"\nFound {len(indices):,} matches{suffix}:")

                rows = []
                for idx in indices:
                    rec_df = explorer.get_record(idx)
                    rows.append(rec_df.iloc[0])
                results = pd.DataFrame(rows)
                results.index = indices
                results._index_type = 'record_number'

                explorer.display_records(results, truncate=not args.full)
                print(f"\nTip: Use --record <number> to view any specific record")

                if args.export:
                    results.to_csv(args.export, index=False)
                    print(f"Exported to {args.export}")
        
        elif args.stats is not None:
            field = args.stats if args.stats else args.field
            stats = explorer.get_statistics(field if field else None)
            explorer.display_statistics(stats)
        
        else:
            explorer.interactive_mode()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
