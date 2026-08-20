# ns_spectral_audit.py
> Proves *why* Newton-Schulz converts a radial-null CE gradient into an anti-radial weight update, by auditing the singular-value spectrum of the real CE gradient on each body matrix.

## What it does
This is the mechanism-closer for the Newton-Schulz (NS) radial finding in the WD-waste / body-norm-ramp investigation. For each transformer body matrix it takes the **real** cross-entropy gradient `G` on a real batch (single-card, full unsharded matrices), does an SVD `G = U diag(sigma) V^T`, and compares the **sigma-weighted** radial dot `<G,W> = sum_i sigma_i a_i` (where `a_i = u_i^T W v_i`) against the **unweighted** polar dot `<UV^T,W> = sum_i a_i`. The claim it tests: the weighted sum is ~0 (raw CE gradient is radial-null) but the unweighted sum is < 0, because NS flattens the singular values so the unweighted contribution dominates — yielding an anti-radial update from a null gradient. It also traces `cos(X, W)` after each NS iteration (raw -> NS1..NS5), expecting a monotonic climb toward roughly -0.0128.

Part of the Newton-Schulz mechanism family. Siblings in the same `tools/` directory: `ns_radial_isolation.py`, `keel_radial_probe.py`, `normuon_update_decomp.py`, `wd_waste_probe.py`, `wd_waste_partb.py`, `finite_rescale_probe.py`.

## Usage
```bash
# Minimal: audit a checkpoint, resolving its data groups from the config
python ns_spectral_audit.py --ckpt <ckpt.pt> --config <run_config.yaml>

# Audit more matrices, longer sequence, and dump results to JSON
python ns_spectral_audit.py --ckpt <ckpt.pt> --config <run_config.yaml> \
    --n 24 --seq 2048 --out audit.json

# Override the data groups explicitly (comma-separated)
python ns_spectral_audit.py --ckpt <ckpt.pt> --groups groupA,groupB
```

## Key arguments
| flag | default | meaning |
| --- | --- | --- |
| `--ckpt` | (required) | Checkpoint to audit; passed through `resolve_ckpt()` (accepts a run/ckpt reference, not only a raw path). |
| `--config` | `None` | Run config YAML; used by `_resolve_own_groups()` to find the data groups for the panel batch. |
| `--groups` | `None` | Comma-separated data-group override; when set, bypasses config-based group resolution. |
| `--n` | `12` | Number of body matrices to audit (first N matching body params). |
| `--seq` | `1024` | Sequence length for the probe batch; capped at the model's `max_seq_len`. |
| `--nbins` | `5` | Number of singular-value quantile bins (Q0 = highest sigma) for the weighted-vs-unweighted breakdown. |
| `--seed` | `0` | Seed for `build_panel` batch sampling. |
| `--out` | `None` | If set, writes the result summary dict to this path as JSON. |

## Output (format)
Logs (via `logger.print_and_log`, also to `./logs`) and, with `--out`, a JSON dict containing:
medians over audited matrices of the sigma-weighted radial dot, the unweighted polar dot, `cos(raw G, W)`, and `cos(polar, W)`; the per-bin weighted/unweighted contributions (`bin_weighted`, `bin_unweighted`, length `nbins`); and `ns_iter_cos` (median `cos(X,W)` per NS iteration). No dataset text is emitted — only numeric diagnostics.

## Notes
- "Body matrices" are 2-D params ending in `wq/wk/wv/wo.weight` or `w1/w2/w3.weight` (`_isbody`), with a non-None grad.
- The gradient is the **train-branch fused CCE** path: `raw(x, y, active_layers=None, scaffold_mode=False)` then `loss.backward()` — i.e. the same forward path as production, not a reconstruction.
- The NS trace replicates `zeropower_via_newtonschulz5` stage by stage: bf16 cast, transpose if rows > cols, normalize, then one `nsloop_torch` iteration per step (coeffs `a,b,c = 3.4445, -4.7750, 2.0315`).
- Model is loaded full-precision-half (`half_precision=True`) with `shard_strategy="none"` (single-card, full matrices) and `use_keel=None`.
- Imports lean on sibling probes: `resolve_ckpt` and `_resolve_own_groups` from `zloss_row_center_probe.py`, `build_panel` from `rare_token_nll_probe.py`, and `nsloop_torch` from `common_fsdp2/muon_fsdp2.py`. It prepends `../common_fsdp2` and `../saved_code` to `sys.path`, so run it from within `tools/`.
- Reads tokenized data from `cfg.data_root_path` (falling back to a relative `notebooks/datasets/tokenized/llama/` path); needs the checkpoint and its dataset reachable (valhalla NAS / conda `trainenv`).
- Read-only with respect to the checkpoint and data.
