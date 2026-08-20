# wd_waste_partb.py
> Per-matrix `cos(g_loss, W)` probe: measures whether the body-norm ramp is loss-null (cosmetic, WD wasted-but-harmless) or loss-coupled (WD fighting real signal).

## What it does
This is **Part B** of the WD-waste battery (Nexus #167/#169), the decisive follow-up to Part A. Part A established that every body matrix's `||W||` is *ramping* (not at equilibrium) and that weight decay accounts for ~99.8% of weight motion. Part B answers the open question: is that ramp loss-null (the downstream RMSNorm divides the scale out, so growth is cosmetic) or loss-coupled (the loss is actually being dragged by the growth)?

The decisive metric, computed per 2D weight matrix, is the cosine between the **CE-only** gradient and the weight itself:
`cos(g_loss, W) = <g_loss, W> / (||g_loss|| ||W||)`. For a perfectly scale-invariant pre-norm matrix `L(cW)=L(W)`, so `<g_loss, W>=0 => cos=0` and WD's purely radial `-λW` pull is entirely in the loss-null direction (wasted but harmless). A `cos` meaningfully away from 0 means the loss *has* a radial component and WD is fighting real signal. It also reports `wasted_wd_frac = sqrt(1 - cos^2)` and `wd_over_loss = λ||W|| / ||g_loss||` (how much WD outweighs the signal). Read-only: forward + backward on a few real batches, reads `p.grad`, **no optimizer step, no weight mutation, no z-loss/aux folding**. Siblings: the WD-waste analysis (`docs/WD_WASTE_ANALYSIS.md`), the row-center probe (`zloss_row_center_probe.py`, whose loader it reuses), and the rare-token NLL probe (`rare_token_nll_probe.py`, whose `build_panel` it reuses).

## Usage
```bash
# Single-device (4080), default WD 0.02, 4 batches
python wd_waste_partb.py --ckpt <ckpt.pt> --config <run_config.yaml>

# Write JSON results, sharded across rig GPUs
python wd_waste_partb.py --ckpt <ckpt.pt> --config <run_config.yaml> \
    --shard balanced --nbatch 4 --wd 0.02 --out partb_<step>.json

# Override the data groups explicitly (comma-separated)
python wd_waste_partb.py --ckpt <ckpt.pt> --groups groupA,groupB --nbatch 8
```

## Key arguments
| flag | default | meaning |
|------|---------|---------|
| `--ckpt` | (required) | Checkpoint path; resolved via `resolve_ckpt` (valhalla-aware). Step parsed from `_<step>.pt`. |
| `--config` | `None` | Run YAML; used by `_resolve_own_groups` to determine data groups. Pass this or `--groups`. |
| `--groups` | `None` | Comma-separated data-group override; bypasses config-based group resolution. |
| `--nbatch` | `4` | Number of batches to accumulate CE-only gradients over. |
| `--seq` | `2048` | Sequence window; capped at the model's `max_seq_len`. |
| `--wd` | `0.02` | λ used only to compute the reported `wd_over_loss` ratio (not applied to weights). |
| `--shard` | `"none"` | Shard strategy for `load_model_and_tokenizer` (e.g. `balanced` for multi-GPU). |
| `--seed` | `0` | Seed for `build_panel` data sampling. |
| `--out` | `None` | Output JSON path; if omitted, results are only logged, not written. |

## Notes
- **GPU required.** Loads the model in half precision (`half_precision=True`) and runs a real backward. Device auto-detected via `nc.detect_device`.
- Computes plain `cross_entropy` itself (with `ignore_index=pad_id`) on the model's logits to *guarantee* CE-only — it does not trust the model's own loss (which may fold in z-loss/aux). Loss is scaled by `1/nbatch` per backward.
- **DTensor-aware:** for sharded params it operates on local shards and `all_reduce`s `<g,W>`, `||W||^2`, `||g||^2` over each param's mesh group, so the cosine is global. Non-DTensor (single-device) is purely local.
- Only **2D params with a non-None grad** are scored. Matrices are tagged by scale-invariance class via `_classify`: `body_proj_prenorm` (`attention.wo`, `feed_forward.w2`), `body_in_prenorm` (wq/wk/wv, w1/w3), `embedding`, `head` (`output.`), `other`.
- Logs a per-class summary (cos |mean|, median, max|cos|, median wd/loss) plus the top-10 most loss-coupled matrices by `|cos|`. Interpretation: `|cos|~0` => loss-null/survivable; `|cos|` meaningfully `>0` => loss-coupled/harmful.
- Reuses `resolve_ckpt`/`_resolve_own_groups` from `zloss_row_center_probe` and `build_panel` from `rare_token_nll_probe`; expects `common_fsdp2` (and `saved_code`) on the path (added automatically). Writes a log to `./logs/wd_waste_partb_log.txt`.
- The module docstring frames this as Part B; the value `cos = -0.0129` referenced in project memory is the kind of signed body-class result this probe produces.
