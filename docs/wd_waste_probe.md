# wd_waste_probe.py
> Part A of the WD-waste battery: pure log-parse of `diagnostics.jsonl` that estimates how much of each matrix's gradient motion is wasted decoupled-WD "fighting" vs. real loss signal, and tests whether `||W||` is still ramping or has reached equilibrium.

## What it does
Tests Rook's hypothesis (Nexus #167) that body layers — not just the output head — accumulate loss-null "gauge"-like components that decoupled weight decay then wastes gradient fighting, the suspected source of the ever-growing average gradient norm. The key insight: wherever the loss is invariant to a weight transform (softmax shift-invariance on the head, RMSNorm scale-invariance on pre-norm body matrices), the loss-gradient is orthogonal to the radial direction, but decoupled WD (`-lambda*W`) is entirely radial, so its contribution is ~loss-null wasted motion. This is "Part A" and is nearly free — it relies only on per-layer `w_norm` and pre-WD CE `g_norm` already logged in `diagnostics.jsonl`, so there is no GPU work, no checkpoint load, just a log parse. It is the lightweight front-end of the broader WD-waste / body-norm-ramp investigation (see `docs/WD_WASTE_ANALYSIS.md` and the Newton-Schulz radial-mechanism work for the heavier siblings).

For each tracked matrix it computes `wd_grad_mag = lambda * w_norm`, `wd_grad_share = wd_grad_mag / (wd_grad_mag + g_norm)` (WD's fraction of total motion on that weight), with `g_norm` as the real loss signal, then tracks these plus `||W||_F` across training and runs a ramping test on the recent norm slope.

## Usage
```bash
# Basic: parse a run's diagnostics with the default lambda
python wd_waste_probe.py --diag /path/to/run/diagnostics.jsonl

# Match the run's actual weight_decay and dump the full per-step series to JSON
python wd_waste_probe.py --diag /path/to/run/diagnostics.jsonl --wd 0.02 --out report.json
```
The probe prints a report to stdout (summed WD-share trajectory, output-head positive control, body `wd_grad_share` by depth third, top-8 body matrices by share, and the `||W||` ramping test). With `--out` it also writes the full per-step series as JSON.

## Key arguments
| flag | default | meaning |
|------|---------|---------|
| `--diag` | (required) | Path to the run's `diagnostics.jsonl`. |
| `--wd` | `0.02` | The decoupled weight-decay lambda to assume; set this to the run's actual `weight_decay` so `wd_grad_mag = lambda * w_norm` is correct. |
| `--out` | `None` | Optional path to dump the full per-step series as indented JSON. |

## Notes
- Read-only and dependency-light: pure stdlib (`json`, `argparse`), no GPU, no checkpoint, no model load. Safe to run anywhere the `diagnostics.jsonl` is reachable.
- Correctness depends on two confirmed prerequisites in `muon_fsdp2` / `train_mara`: WD is **decoupled** (`p.mul_(1 - eta*lambda)`), and the logged `g_norm` is the loss gradient captured **after backward but before** the optimizer's WD step, so it excludes the `-lambda*W` term and the decomposition is clean. If a run uses coupled WD or logs `g_norm` post-WD, the share numbers are invalid.
- `--wd` is supplied manually; the probe does not read the run config, so a mismatch silently rescales every `wd_grad_mag`.
- Expects `diagnostics.jsonl` records with `step`, `layers` (each with `idx` and `attn`/`ffn` sub-blocks), plus optional `output` and `tok_embeddings` blocks; each tracked block needs `w_norm` and `g_norm` (blocks missing either are skipped). The JSONL reader is tolerant of older files with multiple concatenated JSON objects per line and silently skips unparseable fragments.
- The output head is treated as a positive control (known common-mode gauge accumulator). The ramping test flags body matrices as "flat" when the recent `||W||` slope is `< 0.5%/kstep`, otherwise "still climbing" — flat = benign equilibrium, persistent positive slope = pathological ramp.
