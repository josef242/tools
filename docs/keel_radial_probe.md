# keel_radial_probe.py
> Read-only checkpoint probe that tests WHY pure cross-entropy produces a small anti-radial body gradient (⟨g,W⟩<0, which grows ‖W‖) in the KEEL highway, via train-vs-eval gradient, branch-gain derivative, ε-sensitivity, and a bf16/fp32-accum discriminator.

## What it does
This is the mechanism-confirmer for the body-norm-ramp / weight-decay-waste investigation (see `docs/WD_WASTE_ANALYSIS.md`). The working hypothesis is that body norm acts as an implicit BRANCH-GAIN knob in the KEEL residual highway `x_{l+1}=Norm(α·x_l + F_l(Norm(x_l)))`; a negative radial gradient means the model "wants more branch relative to highway." It runs offline against a saved checkpoint with no optimizer and no persisted weight mutation (all finite-difference / ε perturbations are applied then restored).

Two modes:
- **`legacy`** (default) — three probes on one load: (1) signed cos(g,W) per body matrix under both `model.eval()` and `model.train()` forward paths, plus a central finite-difference check that the anti-radial gradient is a real loss derivative; (2) branch-gain derivative dL/d log g_l from a per-block forward-hook scalar gain; (3) RMSNorm-ε sensitivity sweep.
- **`stageb`** — corrected train-branch fused-CCE vs eval-branch external-CE comparison across context lengths, with a bf16-accum vs fp32-accum discriminator (tests whether the lean is a bf16 fused-CCE accumulation artifact). Optionally replays exact captured tokens (`--tokens-file`) to split the residual into real-stream-DATA vs FSDP/bf16-PATH.

Sibling probes referenced/imported: `zloss_row_center_probe.py` (`resolve_ckpt`, `_resolve_own_groups`) and `rare_token_nll_probe.py` (`build_panel`); related to the broader Newton-Schulz / z-loss / WD-waste battery.

## Usage
```bash
# legacy: original 3 probes (eval-vs-train cos, branch-gain, eps sweep)
python keel_radial_probe.py --ckpt mf_14000.pt --groups "mf" --out k.json

# stageb: train-CCE vs eval-CE + bf16/fp32-accum discriminator across context lengths
python keel_radial_probe.py --ckpt mf_14000.pt --groups "mf" --mode stageb \
    --seqs 1024,12288 --dtype bf16 --out kb.json

# stageb replay: feed exact captured tokens (opaque) instead of build_panel
python keel_radial_probe.py --ckpt mf_14000.pt --mode stageb \
    --tokens-file captured_tokens.pt --out kb_replay.json
```

## Key arguments
| flag | default | meaning |
| --- | --- | --- |
| `--ckpt` | (required) | checkpoint path/name; resolved via `resolve_ckpt`. Step parsed from `_<N>.pt`. |
| `--groups` | `None` | comma-separated data-group names for `build_panel`; overrides config-derived groups. |
| `--config` | `None` | config path used to resolve groups when `--groups` absent. |
| `--mode` | `legacy` | `legacy` = original 3 probes; `stageb` = train-CCE vs eval-CE + accum discriminator. |
| `--seq` | `1024` | context length (legacy; and stageb fallback when `--seqs` absent). |
| `--seqs` | `None` | stageb only: comma-separated context lengths, e.g. `1024,12288`. |
| `--dtype` | `bf16` | stageb only: autocast dtype (`bf16`/`fp16`/`fp32`); match the run's data_type. |
| `--tokens-file` | `None` | stageb only: replay exact captured tokens (raw binary `{'x','y'}` from WD_DUMP_TOKENS) instead of `build_panel`; overrides `--seqs` with the captured T. |
| `--shard` | `none` | shard strategy passed to the model loader. |
| `--ntokens` | `2048` | declared but not used in the active code paths. |
| `--seed` | `0` | seed for `build_panel` token sampling. |
| `--out` | `None` | write the result dict to this JSON path. |

## Notes
- Loads with `half_precision=True`; legacy probes run grads directly, stageb wraps each loss path in `torch.autocast` at `--dtype`.
- DTensor-aware: `_local`/`_gsum` reduce dot/norms across the device mesh when distributed is initialized, so it works on sharded checkpoints (default `--shard none` is single-device).
- "Body matrices" = params ending in `wq/wk/wv/wo/w1/w2/w3.weight` (`_isbody`); norms, embeddings, and head are excluded.
- stageb forces per-block `use_activation_checkpointing=True` to fit long context and faithfully match the production backward graph; it also forces `raw.train()` (dropout=0 so the backbone is identical).
- The `train_cce_f32accum` path captures `h` after `raw.norm` via a forward hook and recomputes `model_v2.cce_loss` with `accum_e_fp32=accum_c_fp32=True` — same kernel/h, only accum dtype differs. Verdict logic: train_cce_bf16 << 0 but eval_ce ~ 0 ⟹ kernel source; train_cce_f32accum ~ eval_ce (~0) ⟹ bf16-accum artifact; train_cce_f32accum still << 0 ⟹ real fused-CE geometry.
- Token replay treats tokens as an opaque int black box — loaded and fed straight to the model, never decoded, printed, or logged beyond shapes.
- Imports `logger`, `neo_common` (`nc`), and `model_v2` from `../common_fsdp2` / `../saved_code` (added to `sys.path`); writes a log to `./logs/keel_radial_log.txt`. Run from the `tools/` directory so the relative `sys.path` inserts resolve.
- Caveat documented in source: the original legacy PROBE 1 `_ce_loss` always uses the eval branch, so legacy "train vs eval" is effectively the eval-branch path under both modes — stageb was added to exercise the actual train-branch fused-CCE.
