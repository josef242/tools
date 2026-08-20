# gpm.py
> Computes the Gradient Productivity Metric (GPM): do above-trend grad-norm spikes predict above-trend loss improvement on the *next* step?

## What it does
GPM measures whether local grad-norm fluctuations (above the training envelope) predict
local loss improvement on the next batch. It detrends both `nrm` and the step-to-step loss
drop against a rolling median over a window `W`, then Spearman-correlates the residuals — so
it captures whether *local* norm spikes are productive, not the trivial "norm tracks learning"
envelope correlation. Positive GPM => high-norm updates help the next (unseen) batch; ~0 =>
spikes are noise; negative => spikes precede loss increases. It is part of the gradient-
productivity line of the Newton-Schulz / tangent-projection investigation: the tangent-
projected regime shows ~2-3.5x more productive gradients (e.g. KeelHaul vs mf vs DN2 at W=51).
The same trailing-window math here mirrors the in-trainer `GPMTracker`, so retrofitted
historical runs are directly comparable to live runs on the Dashboard.

## Usage
```bash
# Window sweep + over-training breakdown for one run (human-readable)
python gpm.py --log RUN/gen_log.txt

# Compare several runs (names resolved under --root)
python gpm.py --compare keelhaul,dreadnought,mf-low-lr

# Per-step GPM-S / GPM-L matching the LIVE tracker, for the Dashboard
python gpm.py --retrofit RUN/gen_log.txt --out gpm.csv

# Machine-readable summary
python gpm.py --log RUN/gen_log.txt --json
```

## Key arguments
| flag | default | meaning |
|------|---------|---------|
| `--log` | – | One run's `gen_log.txt`; runs centered-window sweep + over-time segments. |
| `--compare` | – | Comma-separated run names; each resolved as `<root>/<name>/gen_log.txt`. |
| `--retrofit` | – | A `gen_log.txt` to replay through the live trailing-window algorithm; pair with `--out`. |
| `--out` | – | Output path for `--retrofit`; `.json` writes JSON records, anything else writes CSV. |
| `--windows` | `5,9,15,25,51,101,201,401` | Window sizes for the centered sweep. |
| `--short` | `15` | Retrofit/live SHORT trailing window (matches trainer default). |
| `--long` | `101` | Retrofit/live LONG trailing window (matches trainer default). |
| `--lag` | `1` | Step lag: `nrm[N]` vs `dloss[N] = ls[N] - ls[N+lag]`. |
| `--json` | off | Emit machine-readable summary for `--log` / `--compare`. |
| `--root` | `~/brainbox/checkpoints/current` | Base dir for `--compare` run-name lookup. |

## Notes
- Input is a training log line containing `st:`, `ls:`, and `nrm:` fields (parsed by regex);
  steps missing either `ls` or `nrm` are dropped, and rows are sorted and deduped by step.
- Two detrend forms: the sweep/compare modes use a **centered** ±`W//2` rolling-median residual
  (`gpm_centered`); retrofit uses a **trailing** window via `GPMReplay`/`gpm_window` so it is
  bit-comparable to the live `GPMTracker`. Don't mix the two when comparing numbers.
- Spearman (rank) correlation is used deliberately for robustness to the heavy-tailed norm
  spikes that are the point of the metric; needs `n >= 4` per window or returns `None`.
- CSV output columns: `step,ls,nrm,gpm_s,gpm_l` (empty cells where GPM is `None`, e.g. early
  steps before the window fills). The `--retrofit` print reports how many steps had a GPM-L.
- Pure stdlib (`re`, `argparse`, `json`, `statistics`, `collections.deque`) — no GPU, no
  checkpoint load, no extra deps; reads logs only.
- Window is its own experiment: too small => batch noise => GPM ~ 0; too large => envelope
  tautology => GPM -> 1. The short-vs-long gap is itself a signal (S>L rising, S<L falling).
