# gpm_retrofit_inject.py
> Splices the live-trainer `| gpm: +S/+L` status field into a historical `gen_log.txt` in place, byte-identical to what the in-trainer GPMTracker emits going forward.

## What it does
Older training runs (e.g. dreadnought/DN2, mf-low-lr) finished before the Gradient Productivity Metric (GPM) field existed, so their `gen_log.txt` step lines lack the `gpm:` field the Dashboard now parses. This tool replays each historical run through `GPMReplay` from `gpm.py` — the same trailing-window math the in-trainer `GPMTracker` uses — and appends the matching `gpm:` field to every training step line, so retrofitted and future live runs are computed identically and are directly comparable on the Dashboard.

It is the retrofit/edit-in-place companion to `gpm.py`, which owns the GPM math and offers analysis modes (`--log` window sweep, `--compare` across runs, `--retrofit` to emit a separate per-step CSV/JSON). This tool instead edits the log file itself. GPM is part of the gradient-productivity strand of the KEEL/NorMuon investigation (Spearman correlation of detrended grad-norm spikes vs. detrended next-step loss drop).

## Usage
```bash
# In place, default windows (short=15, long=101, lag=1)
python gpm_retrofit_inject.py PATH/gen_log.txt

# Preview only — print counts and a sample, write nothing
python gpm_retrofit_inject.py PATH/gen_log.txt --dry-run

# Custom windows
python gpm_retrofit_inject.py PATH/gen_log.txt --short 15 --long 101 --lag 1
```

## Key arguments
| flag | default | meaning |
|------|---------|---------|
| `log` (positional) | — | Path to `gen_log.txt`, edited in place. |
| `--short` | `15` | Trailing SHORT window length (matches trainer default). |
| `--long` | `101` | Trailing LONG window length (matches trainer default). |
| `--lag` | `1` | Lag for the dloss term: `nrm[N]` vs `ls[N]-ls[N+lag]`. |
| `--dry-run` | off | Print what would change; do not write. |

## Notes
- Imports `GPMReplay` from `gpm.py` in the same `tools/` directory (added to `sys.path` at runtime); that file is the single source of truth for the window math. No torch/GPU needed — pure text + stdlib.
- Only edits TRAINING step lines, identified by the Dashboard's `pattern_mara` (regex requiring `st:`, `ls:`, `nrm:`, and `t_tk:`). Eval/avg lines, headers, and blank lines are copied verbatim.
- Idempotent: a line that already has `gpm:` is left unchanged, but its `(nrm, ls)` is still pushed into the replay buffer so the trailing windows stay aligned. Safe to re-run.
- For the first <5 steps both windows return None, so no tag is appended — matching the live tracker, whose `status_tag` returns `""` in that case.
- Field format replicates `GPMTracker.status_tag` exactly: ` | gpm: +0.31/+0.25`, value formatter `f"{v:+.2f}"` (or ` -- ` when one side is None). No trend arrow (dropped 2026-06-24).
- Safety: writes a timestamped `<path>.pre-gpm.<YYYYMMDD-HHMMSS>.bak` backup, then does an atomic replace via a `.tmp` file + `os.replace`. Original line endings (`\r\n` vs `\n`) are preserved per line.
- Console output reports total lines, training lines, count injected, count already had `gpm:`, the window params, and a trailing sample of one tagged line.
