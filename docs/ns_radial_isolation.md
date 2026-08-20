# ns_radial_isolation.py
> Isolated, FSDP-free probe that pushes a radial-null gradient through the real Muon stages (Newton-Schulz → scaling → NorMuon) to find which stage injects the anti-radial (≈ −0.0129) lean.

## What it does
Settles whether the small anti-radial component seen in the body-norm / weight-decay-waste
investigation comes from the **math of Newton-Schulz orthogonalization** (intrinsic to any correct
implementation) or from a **downstream stage of this specific NorMuon implementation**
(`apply_scaling` / `apply_normuon`). It builds a random gradient `G` that is exactly orthogonal to
the weight `W` (radial-null, `⟨G,W⟩=0`), then runs it through the real `muon_fsdp2` stages one at a
time, measuring `cos(stage_output, W)` after each. If the cosine jumps strongly negative right after
Newton-Schulz, the lean is intrinsic to NS; if it stays ~0 through NS and only drops at scaling or
NorMuon, that later stage is the source. It runs pure single-card fp32 math with no trainer, no
FSDP, and no data — it only loads real body `W` matrices from a checkpoint for realistic shapes and
scale. Part of the Newton-Schulz radial-mechanism family (see MEMORY: "Newton-Schulz radial finding")
and the broader WD-waste battery.

## Usage
```bash
# Minimal: probe a checkpoint, default 40 body matrices
python ns_radial_isolation.py --ckpt <model_*.pt or run_dir>

# Larger sample, fixed seed, dump JSON summary
python ns_radial_isolation.py --ckpt /path/to/run_dir --n 80 --seed 0 --out ns_radial.json

# Vary Newton-Schulz iteration count / NorMuon beta2
python ns_radial_isolation.py --ckpt <ckpt> --ns-steps 5 --beta2 0.99
```
`--ckpt` accepts either a direct `model_*.pt` file or a run directory; if given a directory,
`resolve_ckpt` picks the highest-step `model_<N>.pt` in it.

## Key arguments
| flag | default | meaning |
| --- | --- | --- |
| `--ckpt` | (required) | Checkpoint `.pt` file, or a run dir (auto-selects highest-step `model_*.pt`). |
| `--n` | `40` | Number of body matrices to sample (first N of the matched bodies). |
| `--seed` | `0` | Seed for the per-matrix random gradient generator. |
| `--ns-steps` | `5` | Newton-Schulz iteration count passed to `zeropower_via_newtonschulz5`. |
| `--beta2` | `0.99` | NorMuon second-moment decay used by `apply_normuon`. |
| `--out` | `None` | If set, write the per-stage summary dict as JSON to this path. |

## Output
Prints `cos(stage, W)` summary stats (median, mean, neg-fraction, n) for four stages:
`rawG` (≈0 by construction), `afterNS`, `afterScale`, `afterNormuon`. The same dict is returned and
optionally written to `--out` as JSON. No model weights or data text are emitted.

## Notes
- **Body-matrix filter**: only 2-D params whose names end in `wq/wk/wv/wo/w1/w2/w3.weight` are
  sampled (attention + FFN projections); norms, embeddings, and the output head are excluded.
- **Dependencies**: imports from `../common_fsdp2` (`neo_common`, `muon_fsdp2`'s
  `zeropower_via_newtonschulz5`, `apply_scaling`, `apply_normuon`, plus `logger`) and reuses
  `resolve_ckpt` from `zloss_row_center_probe.py` in this `tools/` dir — those must be importable.
- **NorMuon second moment is cold**: `apply_normuon` is fed a zeros second-moment buffer, so this
  tests the rescale *shape* at step 0, not a warmed-up state.
- Model is loaded `half_precision=True` with `shard_strategy="none"` (single-card, no FSDP), but the
  cosine math itself is done in fp32. Writes a `./logs` dir via the shared `logger`.
- `apply_scaling` is called with `rms_scale=False`.
