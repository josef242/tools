# normuon_update_decomp.py
> Decomposes the ACTUAL applied NorMuon step (post Newton-Schulz / RMS / neuron-norm) per body matrix to check whether the normalized update — not the raw gradient — injects the radial drift that feeds the body-norm ramp.

## What it does
This is "Math Agent test #2" in the body-norm-ramp / weight-decay-waste investigation, the follow-up to Part B (`wd_waste_partb.py`, which measured `cos(g_loss, W)` on the raw gradient and found it radial-null). The key insight: NorMuon does not step along the gradient — it steps along the Newton-Schulz-orthogonalized, RMS-scaled, neuron-normalized update. So a radial-null gradient does NOT imply a radial-null step. This tool replicates the exact NorMuon update transform on each body matrix and compares `cos(grad, W)` against `cos(UPDATE, W)`; if the update cosine is much larger, the orthogonalization itself is feeding the ramp — the mechanism a fix must target. Siblings: `wd_waste_partb.py`, `zloss_row_center_probe.py`, `rare_token_nll_probe.py`; canonical writeup in `docs/WD_WASTE_ANALYSIS.md`.

## Usage
```bash
# Single-device (small body matrices fit) — preferred
python normuon_update_decomp.py --ckpt <ckpt.pt> --groups "<group>" --out nud_<tag>.json

# Big model, sharded across devices (NS on row-slices is approximate — flagged)
python normuon_update_decomp.py --ckpt <ckpt.pt> --groups "<g1>,<g2>" --shard balanced

# Resolve eval groups from a run config instead of passing them explicitly
python normuon_update_decomp.py --ckpt <ckpt.pt> --config <run_dir/config.yaml>
```

## Key arguments
| flag | default | meaning |
|------|---------|---------|
| `--ckpt` | (required) | Checkpoint path; resolved via `resolve_ckpt` (accepts shorthand). |
| `--groups` | `None` | Comma-separated data-group names for the CE eval batch; overrides config-derived groups. |
| `--config` | `None` | Run config used to resolve groups when `--groups` is omitted. |
| `--ntokens` | `2048` | Total tokens used to compute the CE-only gradient. |
| `--seq` | `1024` | Sequence window length (clamped to model `max_seq_len`). |
| `--shard` | `none` | Load shard strategy; `none` = single-device (exact NS), `balanced` = sharded big-model (row-slice NS, approximate). |
| `--wd` | `0.02` | Weight-decay value for the decoupled-WD bookkeeping term (analytic). |
| `--seed` | `0` | Seed for the eval-batch panel build. |
| `--out` | `None` | Output JSON path; if omitted, results are logged only, not written. |

## Output (format only)
Returns / writes a JSON dict with `checkpoint`, `step`, `mode` (`normuon_transform`), a `summary` keyed by matrix class (`body_proj_prenorm`, `body_in_prenorm`, `head`, `embedding`) and a `per_matrix` list. Per-matrix fields include `name`, `cls`, `w_norm`, `cos_grad_W`, `cos_muonupd_W`, `update_norm`. Summary fields include grad/update cosine abs-mean, median, max-abs, and `update_radial_meancos` (net signed radial push of the update). No dataset text is emitted — only parameter names and numeric geometry.

## Notes
- GPU tool; `nc.detect_device` selects the device, model loaded in half precision via `neo_common.load_model_and_tokenizer`.
- It does NOT actually call `optimizer.step()` — despite the docstring's "one real step" framing, the code replicates the NorMuon transform (`apply_momentum` → `zeropower_via_newtonschulz5` → `apply_scaling` → `apply_normuon`) directly from `muon_fsdp2`, with COLD momentum and second-moment buffers (so magnitudes shift from a warm run, but the radial cosine is geometry-dominated by NS — flagged in the source). Weights are not mutated and nothing is saved.
- Transform hyperparameters (`ns_steps`, `momentum`, `beta2`, `rms_scale`) are read from the checkpoint's config with defaults `5 / 0.95 / 0.95 / False`.
- Only 2D params with a non-None grad are processed (body/head/embedding matrices).
- `--shard balanced`: under sharded DTensor loads each rank holds a row-slice; Newton-Schulz on a partial matrix is an approximation. Prefer `--shard none` when the body matrices fit on one device.
- Depends on sibling tools in `tools/`: `resolve_ckpt` / `_resolve_own_groups` (`zloss_row_center_probe`), `build_panel` (`rare_token_nll_probe`), `_classify` (`wd_waste_partb`), plus `neo_common` and `logger` from `../common_fsdp2`. Logs to `./logs/normuon_update_decomp_log.txt`.
