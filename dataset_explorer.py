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
from typing import Optional, Dict, Any, List, Union, Tuple
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
                if dirpath.exists():
                    shutil.rmtree(dirpath)
                    print(f"Cleaned up temporary directory: {dirpath}")
            except Exception as e:
                print(f"Warning: Could not delete temporary directory {dirpath}: {e}")


# Global temporary file manager
temp_manager = TemporaryFileManager()


def decompress_zst_file(zst_filepath: Path, console: Optional[Console] = None) -> Path:
    """
    Decompress a .zst file to a temporary location.
    Returns the path to the decompressed file.
    """
    if not ZSTD_AVAILABLE:
        raise ImportError(
            "zstandard library not installed. Please install it to work with .zst files:\n"
            "pip install zstandard"
        )
    
    # Get file size for progress tracking
    file_size = zst_filepath.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    # Create temporary directory in the same location as the source file
    # This works on both Unix and Windows
    temp_dir = zst_filepath.parent / "tmp"
    temp_dir.mkdir(exist_ok=True)
    temp_manager.register_dir(temp_dir)
    
    # Use hash to avoid conflicts with multiple files
    file_hash = hashlib.md5(str(zst_filepath.absolute()).encode()).hexdigest()[:8]
    decompressed_name = zst_filepath.stem  # Remove .zst extension
    temp_filepath = temp_dir / f"{decompressed_name}_{file_hash}"
    
    # Check if already decompressed
    if temp_filepath.exists():
        # Verify it's still valid
        original_mtime = zst_filepath.stat().st_mtime
        temp_mtime = temp_filepath.stat().st_mtime
        
        if temp_mtime > original_mtime:
            if console and RICH_AVAILABLE:
                console.print(f"[green]Using existing decompressed file: {temp_filepath.name}[/green]")
            else:
                print(f"Using existing decompressed file: {temp_filepath.name}")
            return temp_filepath
        else:
            # Original file is newer, re-decompress
            temp_filepath.unlink()
    
    print(f"Decompressing {file_size_mb:.1f} MB .zst file...")
    print(f"Temporary location: {temp_filepath}")
    
    # Register for cleanup
    temp_manager.register_file(temp_filepath)
    
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
                with open(temp_filepath, 'wb') as outfile:
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
            with open(temp_filepath, 'wb') as outfile:
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
                    
                    if chunk_count % 100 == 0:  # Update every 100MB
                        print(f"  Processed {bytes_written/(1024*1024):.1f} MB...")
        
        decompressed_size_mb = bytes_written / (1024 * 1024)
        print(f"Decompressed size: {decompressed_size_mb:.1f} MB (ratio: {decompressed_size_mb/file_size_mb:.1f}x)")
    
    print(f"Decompression complete: {temp_filepath.name}")
    return temp_filepath


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


def convert_json_array_to_jsonl(json_filepath: Path, console: Optional[Console] = None) -> Path:
    """Convert a top-level JSON array file to a JSONL temp file. Returns its path.

    Loads the array fully via json.load (memory cost ≈ 2-3x file size while parsing).
    Caches the converted file in <source_dir>/tmp/ keyed on the original path; reuses
    if newer than the source.
    """
    file_size = json_filepath.stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    temp_dir = json_filepath.parent / "tmp"
    temp_dir.mkdir(exist_ok=True)
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
    with open(temp_filepath, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Conversion complete: {temp_filepath.name}")
    return temp_filepath


class MetadataCache:
    """Handles caching and retrieval of dataset metadata."""
    
    def __init__(self, data_filepath: Path, original_filepath: Optional[Path] = None):
        """
        Initialize cache for a data file.
        
        Args:
            data_filepath: Path to the actual data file (may be decompressed temp file)
            original_filepath: Path to the original file if data_filepath is a temp file
        """
        self.data_filepath = data_filepath
        self.original_filepath = original_filepath
        
        # Use original filepath for cache if this is a decompressed file
        cache_filepath = original_filepath if original_filepath else data_filepath
        self.cache_dir = cache_filepath.parent / '.dataset_explorer_cache'
        self.cache_dir.mkdir(exist_ok=True)
        
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


class DatasetExplorer:
    """Main class for exploring large dataset files."""
    
    def __init__(self, filepath: str, max_display_width: int = 200, quick_mode: bool = False,
                 no_cache: bool = False, rebuild_cache: bool = False):
        self.original_filepath = Path(filepath)
        if not self.original_filepath.exists():
            raise FileNotFoundError(f"File or directory not found: {filepath}")

        self.max_display_width = max_display_width
        self.quick_mode = quick_mode
        self.no_cache = no_cache
        self.rebuild_cache = rebuild_cache
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
            self._init_multi_file()
        else:
            self._init_single_file()

    def _init_single_file(self):
        """Set up the explorer for a single source file."""
        self.filepath = self.original_filepath  # May be replaced with a temp file

        if self.original_filepath.suffix.lower() == '.zst':
            self.is_compressed = True
            if '.jsonl.zst' in self.original_filepath.name.lower():
                self.filepath = decompress_zst_file(self.original_filepath, self.console)
                self.file_type = 'jsonl'
            else:
                raise ValueError(
                    f"Unsupported compressed file type: {self.original_filepath.suffix}\n"
                    f"Currently only .jsonl.zst files are supported."
                )
        else:
            self.file_type = self._detect_file_type()
            if self.file_type == 'jsonl' and _peek_first_nonws_char(self.original_filepath) == '[':
                self.is_json_array = True
                self.filepath = convert_json_array_to_jsonl(self.original_filepath, self.console)
                self.file_type = 'jsonl'

        cache_orig = self.original_filepath if self.filepath != self.original_filepath else None
        self.cache = None if self.no_cache else MetadataCache(self.filepath, cache_orig)

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

    SUPPORTED_EXTENSIONS = ('.parquet', '.jsonl', '.json', '.zst')

    def _list_directory_files(self) -> List[Path]:
        """Top-level supported files in the directory, alphabetically sorted."""
        return sorted(
            p for p in self.original_filepath.iterdir()
            if p.is_file() and (
                p.suffix.lower() in ('.parquet', '.jsonl', '.json')
                or p.name.lower().endswith('.jsonl.zst')
            )
        )

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
                working_path = decompress_zst_file(source_path, self.console)
                file_type = 'jsonl'
            else:
                raise ValueError(
                    f"Unsupported compressed file: {source_path.name}. "
                    f"Only .jsonl.zst is supported."
                )
        elif suffix == '.parquet':
            file_type = 'parquet'
        elif suffix in ('.jsonl', '.json'):
            file_type = 'jsonl'
            if _peek_first_nonws_char(source_path) == '[':
                is_json_array = True
                working_path = convert_json_array_to_jsonl(source_path, self.console)
        else:
            raise ValueError(f"Unsupported file type: {source_path.name}")

        cache_orig = source_path if working_path != source_path else None
        cache = None if self.no_cache else MetadataCache(working_path, cache_orig)
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

        if file_type == 'parquet':
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

        canonical_columns: Optional[List[str]] = None
        canonical_schema: Optional[Dict[str, str]] = None
        canonical_type: Optional[str] = None

        for i, source_path in enumerate(candidates, 1):
            if self.console and RICH_AVAILABLE:
                self.console.print(f"\n[bold cyan]\\[{i}/{len(candidates)}][/bold cyan] {source_path.name}")
            else:
                print(f"\n[{i}/{len(candidates)}] {source_path.name}")

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
    
    def _load_metadata(self):
        """Load basic metadata about the file."""
        # Try to load from cache first
        if self.cache and not self.quick_mode:
            cache_data = self.cache.load()
            if cache_data:
                self.metadata = cache_data['metadata']
                self.line_positions = cache_data.get('line_positions')
                
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
        # Store both original and working file sizes for compressed/converted files
        if self.is_compressed:
            self.metadata['original_file_size'] = self.original_filepath.stat().st_size / (1024 * 1024)  # MB
            self.metadata['decompressed_file_size'] = self.filepath.stat().st_size / (1024 * 1024)  # MB
            self.metadata['file_size'] = self.metadata['decompressed_file_size']  # For compatibility
            self.metadata['compression_ratio'] = self.metadata['decompressed_file_size'] / self.metadata['original_file_size']
        elif self.is_json_array:
            self.metadata['original_file_size'] = self.original_filepath.stat().st_size / (1024 * 1024)  # MB
            self.metadata['converted_file_size'] = self.filepath.stat().st_size / (1024 * 1024)  # MB
            self.metadata['file_size'] = self.metadata['converted_file_size']
        else:
            self.metadata['file_size'] = self.filepath.stat().st_size / (1024 * 1024)  # MB

        self.metadata['file_path'] = str(self.original_filepath.absolute())
        self.metadata['is_compressed'] = self.is_compressed
        self.metadata['is_json_array'] = self.is_json_array
        
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
    
    def get_record(self, index: int) -> pd.DataFrame:
        """Get a specific record by (global) index."""
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

    def find_all_records(self, query: Union[str, List[str]],
                         field: Optional[str] = None,
                         regex: bool = False,
                         limit: Optional[int] = None) -> List[int]:
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

            def line_matches(record: Dict[str, Any]) -> bool:
                for i in range(len(terms)):
                    if not _term_in_record(record, i):
                        return False
                return True

            total_size = sum(p.stat().st_size for _, p, _, _ in file_iter)
            scanned_bytes = 0

            class _LimitHit(Exception):
                pass

            def scan_one(working_path: Path, base_offset: int,
                         progress=None, task=None) -> int:
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
                        if progress is not None and task is not None and line_num % update_every == 0:
                            progress.update(task, completed=scanned_bytes, matches=len(indices))
                return line_num

            if RICH_AVAILABLE and self.console and total_size:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TextColumn("matches: {task.fields[matches]:,}"),
                    TextColumn("file: {task.fields[file_label]}"),
                    TimeRemainingColumn(),
                    console=self.console,
                ) as progress:
                    task = progress.add_task("Scanning records...", total=total_size,
                                             matches=0, file_label="")
                    try:
                        for fidx, working_path, base_offset, _ in file_iter:
                            label = (
                                self.source_files[fidx].name
                                if self.is_directory
                                else working_path.name
                            )
                            progress.update(task, file_label=f"[{fidx + 1}/{len(file_iter)}] {label}")
                            scan_one(working_path, base_offset, progress, task)
                    except _LimitHit:
                        pass
                    progress.update(task, completed=total_size, matches=len(indices))
            else:
                print("Scanning records...")
                try:
                    for fidx, working_path, base_offset, _ in file_iter:
                        if self.is_directory:
                            print(f"  [{fidx + 1}/{len(file_iter)}] {self.source_files[fidx].name}")
                        scan_one(working_path, base_offset)
                except _LimitHit:
                    pass
                print(f"  Done. {len(indices):,} matches.")

            return indices

        return indices

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

        first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
        if len(first_line) > width:
            first_line = first_line[:max(width - 3, 1)] + "..."
        return first_line

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

        print("Commands: info, sample, record, findall, list, goto, stats, export, cache, help, quit")
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
                    print("  findall -n <N> <query>     - Cap at N matches (fast peek; 0 = unlimited)")
                    print("  findall -a t1 t2 ...       - AND match: every term must be present")
                    print("                               (quote terms with spaces, e.g. -a \"foo bar\" baz)")
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
                
                elif command.startswith('findall'):
                    args_str = raw_command[len('findall'):].strip()
                    try:
                        tokens = shlex.split(args_str, posix=True)
                    except ValueError as e:
                        print(f"Could not parse arguments (unbalanced quotes?): {e}")
                        continue
                    field: Optional[str] = None
                    use_regex = False
                    set_full = False
                    match_all = False
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
                        elif tok in ('-r', '--regex'):
                            use_regex = True
                            i += 1
                        elif tok in ('-a', '--all'):
                            match_all = True
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
                            query_arg, field=field, regex=use_regex, limit=limit
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
                    self.search_state = {
                        'indices': match_indices,
                        'cursor': 0,
                        'query': terms if len(terms) > 1 else terms[0],
                        'field': field,
                        'regex': use_regex,
                        'limit_hit': limit_hit,
                    }
                    suffix = " (--limit hit, more may exist)" if limit_hit else ""
                    print(f"Found {len(match_indices):,} matches{suffix}. "
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
        
        Note: .zst files are decompressed to a 'tmp' directory in the same location 
              as the source file for cross-platform compatibility (Windows/Unix).
              Install zstandard: pip install zstandard
        ''')
    )
    
    parser.add_argument('file', help='Path to a data file (.parquet, .jsonl, .json, .jsonl.zst) '
                        'OR a directory of same-format files (alphabetical, top-level only)')
    parser.add_argument('--info', action='store_true', help='Show file info and exit')
    parser.add_argument('--quick', action='store_true', help='Quick mode - estimate counts for large files')
    parser.add_argument('--no-cache', action='store_true', help='Disable metadata caching')
    parser.add_argument('--rebuild-cache', action='store_true', help='Rebuild cache with full index')
    parser.add_argument('--sample', type=int, metavar='N', help='Sample N records')
    parser.add_argument('--record', type=int, metavar='NUMBER', help='Get specific record by number')
    parser.add_argument('--random', action='store_true', help='Random sampling')
    parser.add_argument('--full', action='store_true', help='Show full records without truncation')
    parser.add_argument('--search', type=str, help='Find records matching query (use --field to scope, --limit to cap)')
    parser.add_argument('--field', type=str, help='Specific field for search/stats')
    parser.add_argument('--stats', nargs='?', const='', help='Show statistics (optionally for specific field)')
    parser.add_argument('--export', type=str, help='Export results to file')
    parser.add_argument('--limit', type=int, default=10,
                        help='Max matches to return for --search (default: 10; 0 = unlimited full scan)')
    
    args = parser.parse_args()
    
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
            rebuild_cache=args.rebuild_cache
        )
        
        if args.info:
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
