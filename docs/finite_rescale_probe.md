# finite_rescale_probe.py
> Tests whether scaling a whole weight *class* by a finite factor `c` leaves the forward output (CE) unchanged — the finite-`c` analog of the `cos(g,W)≈0` radial-null result.

## What it does
Part of the WD-waste / body-norm-ramp investigation (see `docs/WD_WASTE_ANALYSIS.md`). The `cos(g_loss, W)≈0` finding only proves the *infinitesimal* radial direction is loss-null; it does **not** prove `L(cW)=L(W)` for a finite `c`. This probe answers the finite question directly: for each weight class it temporarily multiplies that class's matrices by `c`, re-runs the same fixed batch, and measures how far CE (and logits / hidden-stream RMS) drift from the `c=1` baseline. The verdict it prints is the "safe range" — the largest deviation `|1-c|` for which `dCE_rel < 1e-3` per class — used to decide whether body renorm is a usable gentle-control lever or off the table. Siblings (imported helpers): `zloss_row_center_probe.py` (`resolve_ckpt`, `_resolve_own_groups`, `capture_final_h`) and `rare_token_nll_probe.py` (`build_panel`).

It is **forward-only**: no backward, no optimizer, no grad. The rescale is applied in-place under `no_grad` on the local shard (`W *= c`) and restored after each measurement (`W /= c`), so the model is unchanged at exit.

## Usage
```bash
# default: all classes (wo,w2,wq,wk,wv,w1,w3,head) at c in {0.95,0.9,0.8,0.7}
python finite_rescale_probe.py --ckpt <ckpt.pt> --groups "<g1,g2>" --out fr_<tag>.json

# also test the gated-MLP branch (w1 and w3 scaled together)
python finite_rescale_probe.py --ckpt <ckpt.pt> --config <run_dir>/config.yaml --paired --out fr_paired.json

# big model: shard the load; restrict to a couple of classes / scales
python finite_rescale_probe.py --ckpt <ckpt.pt> --groups "<g1>" --shard balanced \
    --classes "wo,w2" --cs "0.99,0.95" --out fr.json
```

## Key arguments
| flag | default | meaning |
|------|---------|---------|
| `--ckpt` | (required) | checkpoint path; resolved via `resolve_ckpt` (step parsed from `_<N>.pt`) |
| `--groups` | `None` | comma-sep data group names for the eval panel; overrides config-derived groups |
| `--config` | `None` | run-dir config used to resolve groups when `--groups` is absent |
| `--ntokens` | `4096` | token budget for the fixed eval panel (`build_panel`) |
| `--cs` | `0.95,0.9,0.8,0.7` | comma-sep scale factors to apply per class |
| `--classes` | `None` (all) | comma-sep subset of classes to test (keys of `CLASS_MATCH`) |
| `--paired` | off | also test class `w1w3` (scale `w1` AND `w3` together — the gated-MLP branch) |
| `--shard` | `none` | shard strategy for model load (`balanced` for large models) |
| `--seed` | `0` | panel sampling seed |
| `--out` | `None` | JSON output path; if omitted, results are logged but not written |

## Notes
- **Classes** matched by param-name suffix (`CLASS_MATCH`): `wo` (`attention.wo.weight`), `w2/wq/wk/wv/w1/w3` (analogous), `head` (`output.weight`, used as a sanity control), and `w1w3` (paired, only when `--paired`). A class with no matching params is silently skipped.
- **Per-rescale metrics** in the JSON `rescales` rows: `dCE` (`|CE(c)-CE(1)|`, the headline), `dCE_rel`, `dlogp_y_mean` and `dlogp_y_p99` (per-token NLL shift, mean and 99th pct of `|Δlogp(target)|`), `dhidden_rms_rel` (relative change of final-norm hidden RMS), plus raw `CE`. Output JSON also carries `checkpoint`, `step`, and `baseline` (`CE`, `hidden_rms`).
- **Verdict line**: per class it reports "safe down to c=<worst_safe>" (smallest `c` with `dCE_rel < 1e-3`) or "NOT safe even at c=0.95". Source hypothesis: `body_proj` (`wo`, `w2`) expected safest; gated `w1`/`w3` more cautious.
- Forward uses the **eval branch** (model called without targets → returns logits, or `(logits, None)`); CE/logp computed in-tool, windowed by the model's `max_seq_len`. Loads in **half precision** (`half_precision=True`) and runs `model.eval()`.
- Logs to `./logs/finite_rescale_log.txt`. Adds `../common_fsdp2` and `../saved_code` to `sys.path`; depends on `neo_common`, `logger`, and the two sibling probes — run from the `tools/` dir so the relative imports resolve. Read-only w.r.t. the checkpoint (restores every weight in place).
