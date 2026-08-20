# dataset_explorer.py
> A read-only "flashlight" for poking at large LLM training dataset files (Parquet / JSONL / `.jsonl.zst` / directories) without loading them into memory.

## What it does
Opens a single data file or a directory of same-format files and lets you inspect schema, count records, sample/fetch individual records, search the contents, and compute per-field statistics. It is built for files too large to load whole: JSONL gets a byte-offset line index (cached to disk) for O(1) random access, and `.jsonl.zst` / top-level-JSON-array inputs are transparently decompressed/converted into a local `tmp/` working file. This is a general dataset-inspection utility (not part of a specific optimizer/mechanism investigation); reach for it when you need to eyeball or grep a training shard.

It runs an interactive REPL by default, but also supports one-shot CLI actions (`--info`, `--sample`, `--record`, `--search`, `--stats`).

## Usage
```bash
# Interactive mode (default) — opens a REPL with caching
python dataset_explorer.py data.parquet
python dataset_explorer.py huge_data.jsonl.zst
python dataset_explorer.py /path/to/shard_dir   # directory of same-format files

# One-shot: show schema/counts and exit
python dataset_explorer.py data.jsonl.zst --info

# Quick mode: estimate (don't fully count) large JSONL
python dataset_explorer.py huge_data.jsonl --quick

# Sample 10 random records (uses cached index if present)
python dataset_explorer.py data.parquet --sample 10 --random --full

# Search and cap at 25 matches; scope to one field
python dataset_explorer.py data.jsonl --search "needle" --field text --limit 25
```

## Key arguments
| flag | default | meaning |
| --- | --- | --- |
| `file` (positional) | required | Path to a `.parquet` / `.jsonl` / `.json` / `.jsonl.zst` file, OR a directory of same-format, same-schema files (top-level, alphabetical). |
| `--info` | off | Print file info (type, sizes, record count, schema, cache/index status) and exit. |
| `--quick` | off | For large JSONL, estimate record count from a sample instead of full count/index. |
| `--no-cache` | off | Disable metadata/index caching. |
| `--rebuild-cache` | off | Clear and rebuild the cache (full line index) on load. |
| `--sample N` | — | Sample N records (combine with `--random`). |
| `--record NUMBER` | — | Fetch one record by 0-based (global) record number. |
| `--random` | off | Random rather than sequential sampling. |
| `--full` | off | Show records untruncated (otherwise truncated at display width). |
| `--search QUERY` | — | Return records containing QUERY (case-insensitive substring). |
| `--field NAME` | — | Restrict `--search`/`--stats` to one field (case-insensitive resolve). |
| `--limit N` | `10` | Max matches for `--search`; `0` = unlimited full scan. |
| `--stats [FIELD]` | — | Dataset stats, or per-field stats if FIELD given. |
| `--export FILE` | — | Write the action's result to FILE (`.csv`/`.json`/`.parquet` by extension). |

### Interactive REPL
With no action flag you get a prompt (`> `). Notable commands: `info`, `sample [n] [random|full]`, `record <n> [full]`, `findall <query>` (supports `-f <field>`, `-r` regex, `-a` AND-of-terms, `-n <N>` cap), match navigation `next`/`prev`/`goto <n>`/`list`/`results`, `findrec <byte_pos>`, `stats [field]`, `export ...`, `cache clear|rebuild|info`, display toggles (`full`, `compact`, `maxdisplay <N>`), `help`, `quit`. Type `help` in-session for the full list.

## Notes
- Dependencies: requires `pandas`, `pyarrow`, `numpy` (hard requirement — exits if missing). `zstandard` is needed only for `.jsonl.zst`; `rich` and `tqdm` are optional (nicer progress/formatting, plain-text fallback otherwise).
- Caching: metadata + JSONL line index are stored in a `.dataset_explorer_cache/` dir next to the source, keyed by path hash, and invalidated on mtime/size change. For compressed/converted inputs the cache keys off the original file.
- Working files: `.jsonl.zst` and top-level-JSON-array `.json` inputs are written to a `tmp/` dir beside the source and registered for cleanup at exit (decompression can be multi-GB; conversion uses `json.load`, ~2-3x file size in RAM while parsing).
- Large single JSONL (>500 MB, non-quick, interactive) prompts (stdin) to choose index/count/estimate/skip — so a fully non-interactive run on such a file may block unless `--quick` or a cache is used.
- Directory mode requires all files to share one format and identical column set, or it raises; record numbers and search indices are GLOBAL across the directory (`findrec` byte-position lookup is single-file only).
- Read-only with respect to source data: it never writes back to the dataset; `--export` only writes the sampled/searched subset you request.
- Content caveat: this tool surfaces raw dataset rows/fields verbatim (potentially unfiltered training text). Treat its output as opaque; this doc deliberately shows no sample rows.
