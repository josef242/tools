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

import json
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
        # Store both original and working filepaths
        self.original_filepath = Path(filepath)
        self.filepath = self.original_filepath  # May be modified if compressed
        self.is_compressed = False
        
        self.max_display_width = max_display_width
        self.quick_mode = quick_mode
        self.no_cache = no_cache
        self.console = Console() if RICH_AVAILABLE else None
        
        if not self.original_filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Handle compressed files BEFORE detecting file type
        if self.original_filepath.suffix.lower() == '.zst':
            self.is_compressed = True
            # Check if it's a supported compressed format
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
        
        self.data = None
        self.metadata = {}
        self.line_positions = None  # For JSONL files
        
        # Initialize cache with original filepath if compressed
        if self.is_compressed:
            self.cache = MetadataCache(self.filepath, self.original_filepath) if not no_cache else None
        else:
            self.cache = MetadataCache(self.filepath) if not no_cache else None
        
        if rebuild_cache and self.cache:
            self.cache.clear()
        
        self._load_metadata()
    
    def _detect_file_type(self) -> str:
        """Detect file type from extension."""
        suffix = self.filepath.suffix.lower()
        if suffix == '.parquet':
            return 'parquet'
        elif suffix in ['.jsonl', '.json']:
            return 'jsonl'
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
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
        # Store both original and working file sizes for compressed files
        if self.is_compressed:
            self.metadata['original_file_size'] = self.original_filepath.stat().st_size / (1024 * 1024)  # MB
            self.metadata['decompressed_file_size'] = self.filepath.stat().st_size / (1024 * 1024)  # MB
            self.metadata['file_size'] = self.metadata['decompressed_file_size']  # For compatibility
            self.metadata['compression_ratio'] = self.metadata['decompressed_file_size'] / self.metadata['original_file_size']
        else:
            self.metadata['file_size'] = self.filepath.stat().st_size / (1024 * 1024)  # MB
        
        self.metadata['file_path'] = str(self.original_filepath.absolute())
        self.metadata['is_compressed'] = self.is_compressed
        
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
        """Get a JSONL record using cached byte position (O(1) operation)."""
        if not self.line_positions or index >= len(self.line_positions):
            return None
        
        byte_pos = self.line_positions[index]
        
        with open(self.filepath, 'rb') as f:
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
        """Get a specific record by index."""
        if self.metadata.get('num_rows') is not None:
            if index < 0 or index >= self.metadata['num_rows']:
                raise ValueError(f"Record number {index} out of range. Dataset has {self.metadata['num_rows']} records (0-{self.metadata['num_rows']-1})")
        elif index < 0:
            raise ValueError(f"Record number must be non-negative (got {index})")
        
        if self.file_type == 'parquet':
            parquet_file = pq.ParquetFile(self.filepath)
            
            current_idx = 0
            for i in range(parquet_file.num_row_groups):
                row_group = parquet_file.metadata.row_group(i)
                group_rows = row_group.num_rows
                
                if current_idx <= index < current_idx + group_rows:
                    df = parquet_file.read_row_group(i).to_pandas()
                    local_idx = index - current_idx
                    result = df.iloc[[local_idx]]
                    result._index_type = 'record_number'
                    return result
                
                current_idx += group_rows
            
            df = pd.read_parquet(self.filepath)
            result = df.iloc[[index]]
            result._index_type = 'record_number'
            return result
        
        elif self.file_type == 'jsonl':
            # Use index for O(1) access if available
            if self.line_positions:
                record = self.get_record_by_position(index)
                if record:
                    df = pd.DataFrame([record], index=[index])
                    df._index_type = 'record_number'
                    return df
                else:
                    raise ValueError(f"Could not read record at index {index}")
            
            # Fallback to sequential reading
            with open(self.filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == index:
                        record = json.loads(line)
                        df = pd.DataFrame([record], index=[index])
                        df._index_type = 'record_number'
                        return df
            
            raise ValueError(f"Could not find record at number {index}")
    
    def search_records(self, query: str, field: Optional[str] = None, 
                      max_results: int = 10, regex: bool = False) -> pd.DataFrame:
        """Search for records matching a query."""
        results = []
        
        if self.file_type == 'parquet':
            parquet_file = pq.ParquetFile(self.filepath)
            current_row = 0
            
            for batch in parquet_file.iter_batches(batch_size=10000):
                df = batch.to_pandas()
                df.index = range(current_row, current_row + len(df))
                
                if field:
                    if field in df.columns:
                        if regex:
                            mask = df[field].astype(str).str.contains(query, case=False, na=False, regex=True)
                        else:
                            mask = df[field].astype(str).str.contains(query, case=False, na=False, regex=False)
                        matches = df[mask]
                    else:
                        current_row += len(df)
                        continue
                else:
                    mask = pd.Series([False] * len(df), index=df.index)
                    for col in df.select_dtypes(include=['object']).columns:
                        if regex:
                            col_mask = df[col].astype(str).str.contains(query, case=False, na=False, regex=True)
                        else:
                            col_mask = df[col].astype(str).str.contains(query, case=False, na=False, regex=False)
                        mask = mask | col_mask
                    
                    for col in df.select_dtypes(include=['number']).columns:
                        col_mask = df[col].astype(str).str.contains(query, case=False, na=False, regex=False)
                        mask = mask | col_mask
                    
                    matches = df[mask]
                
                if len(matches) > 0:
                    results.append(matches)
                    if sum(len(r) for r in results) >= max_results:
                        break
                
                current_row += len(df)
        
        elif self.file_type == 'jsonl':
            record_indices = []
            
            # Show progress for searches on indexed files
            if self.line_positions and self.metadata.get('num_rows'):
                total_records = self.metadata['num_rows']
                print(f"Searching {total_records:,} records...")
                
                for idx in range(total_records):
                    if idx % 100000 == 0 and idx > 0:
                        print(f"  Searched {idx:,} records...")
                    
                    record = self.get_record_by_position(idx)
                    if not record:
                        continue
                    
                    found = False
                    if field:
                        if field in record:
                            val = str(record[field])
                            if regex:
                                if re.search(query, val, re.IGNORECASE):
                                    found = True
                            elif query.lower() in val.lower():
                                found = True
                    else:
                        for val in record.values():
                            val_str = str(val)
                            if regex:
                                if re.search(query, val_str, re.IGNORECASE):
                                    found = True
                                    break
                            elif query.lower() in val_str.lower():
                                found = True
                                break
                    
                    if found:
                        results.append(record)
                        record_indices.append(idx)
                        
                        if len(results) >= max_results:
                            break
            
            else:
                # Fallback to sequential search
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        record = json.loads(line)
                        
                        found = False
                        if field:
                            if field in record:
                                val = str(record[field])
                                if regex:
                                    if re.search(query, val, re.IGNORECASE):
                                        found = True
                                elif query.lower() in val.lower():
                                    found = True
                        else:
                            for val in record.values():
                                val_str = str(val)
                                if regex:
                                    if re.search(query, val_str, re.IGNORECASE):
                                        found = True
                                        break
                                elif query.lower() in val_str.lower():
                                    found = True
                                    break
                        
                        if found:
                            results.append(record)
                            record_indices.append(line_num)
                            
                            if len(results) >= max_results:
                                break
        
        if self.file_type == 'parquet':
            if results:
                combined = pd.concat(results).head(max_results)
                combined._index_type = 'record_number'
                return combined
            else:
                return pd.DataFrame()
        else:
            df = pd.DataFrame(results[:max_results])
            if not df.empty:
                df.index = record_indices[:len(df)]
                df._index_type = 'record_number'
            return df
    
    def find_record_number(self, byte_position: int) -> int:
        """Find the exact record number for a given byte position."""
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
            table = Table(title=f"Dataset Info: {self.original_filepath.name}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("File Type", self.file_type.upper())
            
            # Show compression info if applicable
            if self.is_compressed:
                table.add_row("Original Format", self.original_filepath.suffix.upper())
                table.add_row("Compressed Size", f"{self.metadata['original_file_size']:.2f} MB")
                table.add_row("Decompressed Size", f"{self.metadata['decompressed_file_size']:.2f} MB")
                table.add_row("Compression Ratio", f"{self.metadata['compression_ratio']:.1f}x")
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
            
            if self.cache and self.cache.cache_filepath.exists():
                cache_size = self.cache.cache_filepath.stat().st_size / 1024
                table.add_row("Cache Status", f"✓ Cached ({cache_size:.1f} KB)")
            else:
                table.add_row("Cache Status", "✗ Not cached")
            
            # Show temp file location for compressed files
            if self.is_compressed:
                table.add_row("Temp File", str(self.filepath))
            
            self.console.print(table)
            
            schema_table = Table(title="Schema")
            schema_table.add_column("Field", style="cyan")
            schema_table.add_column("Type", style="yellow")
            
            for field, dtype in self.metadata['schema'].items():
                schema_table.add_row(field, dtype)
            
            self.console.print(schema_table)
        else:
            print("\n" + "="*50)
            print(f"Dataset Info: {self.original_filepath.name}")
            print("="*50)
            print(f"File Type: {self.file_type.upper()}")
            
            # Show compression info if applicable
            if self.is_compressed:
                print(f"Original Format: {self.original_filepath.suffix.upper()}")
                print(f"Compressed Size: {self.metadata['original_file_size']:.2f} MB")
                print(f"Decompressed Size: {self.metadata['decompressed_file_size']:.2f} MB")
                print(f"Compression Ratio: {self.metadata['compression_ratio']:.1f}x")
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
            
            if self.cache and self.cache.cache_filepath.exists():
                cache_size = self.cache.cache_filepath.stat().st_size / 1024
                print(f"Cache Status: ✓ Cached ({cache_size:.1f} KB)")
            else:
                print("Cache Status: ✗ Not cached")
            
            # Show temp file location for compressed files
            if self.is_compressed:
                print(f"Temp File: {self.filepath}")
            
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
                df_sample = pd.read_parquet(self.filepath).head(10000)
                stats['sample_size'] = len(df_sample)
                stats['memory_usage_mb'] = df_sample.memory_usage(deep=True).sum() / (1024 * 1024)
                
                type_counts = df_sample.dtypes.value_counts()
                stats['column_types'] = {str(k): v for k, v in type_counts.items()}
                
                null_counts = df_sample.isnull().sum()
                stats['null_counts'] = null_counts[null_counts > 0].to_dict()
        
        elif self.file_type == 'jsonl':
            field_values = []
            record_sizes = []
            
            # Use index for faster sampling if available
            if self.line_positions:
                sample_size = min(10000, len(self.line_positions))
                for i in range(sample_size):
                    record = self.get_record_by_position(i)
                    if record:
                        if i < len(self.line_positions) - 1:
                            record_size = self.line_positions[i+1] - self.line_positions[i]
                        else:
                            # Estimate last record size
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
        print(f"Loaded: {self.original_filepath.name}")
        
        if self.is_compressed:
            print(f"Format: Compressed {self.original_filepath.suffix.upper()}")
            print(f"Working with decompressed temp file in: {self.filepath.parent}")
        
        if self.cache:
            if self.cache.cache_filepath.exists():
                print(f"Cache: {self.cache.cache_filepath.name}")
            else:
                print("Cache: Not yet created")
        
        print("Commands: info, sample, record, search, stats, export, cache, help, quit")
        print("="*60)
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    print("Goodbye!")
                    break
                
                elif command == 'help':
                    print("\nAvailable commands:")
                    print("  info                   - Show dataset information")
                    print("  sample [n] [random]    - Sample n records (default 5)")
                    print("  sample [n] full        - Sample n records without truncation")
                    print("  record <number>        - Show specific record by number (0-based)")
                    print("  record <number> full   - Show specific record without truncation")
                    print("  findrec <byte_pos>     - Find record number at byte position")
                    print("  search <query>         - Search for records")
                    print("  stats [field]          - Show statistics")
                    print("  export <n> <file>      - Export first n records to file")
                    print("  export record <n> <f>  - Export specific record to file")
                    print("  cache clear            - Clear cached metadata")
                    print("  cache rebuild          - Rebuild cache with full index")
                    print("  cache info             - Show cache information")
                    print("  maxdisplay <width>     - Set maximum display width")
                    print("  quit                   - Exit the program")
                
                elif command == 'info':
                    self.print_info()
                
                elif command.startswith('cache'):
                    parts = command.split()
                    if len(parts) < 2:
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
                        self.max_display_width = width
                        print(f"Maximum display width set to {width}.")
                    except ValueError:
                        print(f"Invalid width: {parts[1]}")
                
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
                
                elif command.startswith('search'):
                    parts = command[6:].strip().split()
                    query = ""
                    limit = 10
                    
                    if parts and parts[0].isdigit():
                        limit = int(parts[0])
                        query = ' '.join(parts[1:])
                    else:
                        query = ' '.join(parts)
                    
                    if not query:
                        query = input("Enter search query: ")
                    
                    print(f"Searching for: {query} (limit: {limit})")
                    results = self.search_records(query, max_results=limit)
                    
                    if len(results) > 0:
                        print(f"\nFound {len(results)} matches:")
                        if len(results) > 20:
                            show_all = input(f"Show all {len(results)} results? (y/n, default=n): ").strip().lower() == 'y'
                            if not show_all:
                                results = results.head(20)
                                print("Showing first 20 results...")
                        
                        truncate = input("Truncate long fields? (y/n, default=y): ").strip().lower() != 'n'
                        self.display_records(results, truncate=truncate)
                        print(f"\nTip: Use 'record <number>' to view any specific record in full")
                    else:
                        print("No matches found.")
                
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
    
    parser.add_argument('file', help='Path to parquet, jsonl, or compressed jsonl.zst file')
    parser.add_argument('--info', action='store_true', help='Show file info and exit')
    parser.add_argument('--quick', action='store_true', help='Quick mode - estimate counts for large files')
    parser.add_argument('--no-cache', action='store_true', help='Disable metadata caching')
    parser.add_argument('--rebuild-cache', action='store_true', help='Rebuild cache with full index')
    parser.add_argument('--sample', type=int, metavar='N', help='Sample N records')
    parser.add_argument('--record', type=int, metavar='NUMBER', help='Get specific record by number')
    parser.add_argument('--random', action='store_true', help='Random sampling')
    parser.add_argument('--full', action='store_true', help='Show full records without truncation')
    parser.add_argument('--search', type=str, help='Search for query in records')
    parser.add_argument('--field', type=str, help='Specific field for search/stats')
    parser.add_argument('--stats', nargs='?', const='', help='Show statistics (optionally for specific field)')
    parser.add_argument('--export', type=str, help='Export results to file')
    parser.add_argument('--limit', type=int, default=10, help='Max results for search (default: 10)')
    
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
            results = explorer.search_records(args.search, args.field, args.limit)
            if len(results) > 0:
                print(f"\nFound {len(results)} matches:")
                explorer.display_records(results, truncate=not args.full)
                print(f"\nTip: Use --record <number> to view any specific record")
                
                if args.export:
                    results.to_csv(args.export, index=False)
                    print(f"Exported to {args.export}")
            else:
                print("No matches found.")
        
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
