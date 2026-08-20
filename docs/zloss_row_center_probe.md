# zloss_row_center_probe.py
> Offline, forward-only probe that splits a checkpoint's output head into a vocab-mean (common-mode "gauge") plus a centered remainder to decide whether a large logZ is a CE-invisible offset versus real classifier inflation/collapse.

## What it does
A logit is `z_i = h.w_i`. This probe row-centers the output head: `mu = mean_i(w_i)`, `w_i = mu + w_i_centered`, so `z_i = h.mu + h.w_i_centered`. Because softmax is shift-invariant, the per-token scalar `h.mu` leaves CE, probabilities, sampling, and the CE gradient unchanged, but `logZ = logsumexp` absorbs it directly (`logZ = h.mu + logZ_c`). The probe measures how much of an inflated `logZ` (e.g. dreadnought's ~490 vs a normal ~7-11) is just this CE-invisible common-mode gauge versus a real centered margin, and checks the centered head geometry for low-rank collapse.

It is part of the z-loss / log-partition (logZ) investigation (Rook Nexus threads #139/#144/#146/#156/#157). The field names it emits (`effective_rank_c`, `small_sigma_pN`) deliberately match `common_fsdp2.row_center.centered_geometry()` so the offline series lines up with live telemetry. The actionable finding it supports: the right fix is gauge subtraction (row-centering, function-preserving), NOT a head-norm/head-WD brake — a blunt brake was shown to cause low-rank collapse rather than fix anything.

## Usage
```bash
# Full probe: weight-side metrics + data-side logZ/logZ_c on the model's own training mix
python zloss_row_center_probe.py --ckpt <path-or-dir> --ntokens 8192 --out results.json

# Fast weight-only path: mmap just output.weight, skip the ~23GB full load + forward
python zloss_row_center_probe.py --ckpt <path-or-dir> --weights-only --aux

# Point .pt at valhalla but read group names from the run-dir config (decouples them)
python zloss_row_center_probe.py --ckpt <ckpt.pt> --config <run_dir>/config_xxx.yaml --aux
```

`--ckpt` may be a single `model_*.pt` file or a directory (it then picks the highest-step `model_*.pt`). On a full run the probe loads the model, runs a forward pass to capture the post-final-norm hidden `h`, computes `logZ`/`logZ_c`/`h.mu` distributions, prints a multi-line VERDICT, and optionally writes a JSON result.

## Key arguments
| flag | default | meaning |
| --- | --- | --- |
| `--ckpt` | (required) | Checkpoint `.pt` file or a directory (latest `model_*.pt` is chosen). |
| `--ntokens` | `8192` | Length of the single `[1, N]` val sequence assembled round-robin from val shards. |
| `--aux` | off | Also analyze SCS aux-head weights (`aux_heads.*.linear.weight`). |
| `--out` | `None` | Write the full result dict to this JSON path. |
| `--weights-only` | off | mmap-extract only `output.weight` (+ aux linears); skips full load and forward — gives weight-side metrics in seconds but NO `logZ_c` data metric. |
| `--groups` | `None` | Comma-separated group override to force a fixed common probe set across runs; default is each model's OWN config groups. |
| `--config` | `None` | Explicit ground-truth `config_*.yaml` to read group NAMES from (lets the `.pt` live on valhalla while the config lives in the run dir). Beats auto-detect; `--groups` still wins if both given. |

## Notes
- Group resolution order: `--groups` override > explicit `--config` yaml > loaded `cfg.groups` > latest `config_*.yaml` beside the ckpt. If none yields names the probe FAILS LOUD rather than silently probing a foreign/empty mix (which would corrupt cross-run `logZ_c` comparisons). Only group NAMES are used — schedule/proportions are intentionally dropped, since `build_val_batch` samples by name.
- Data root comes from `cfg.data_root_path` (default `../../notebooks/datasets/tokenized/llama/`); it looks for `*_val_*.npy` shards under each group dir. The probe is read-only and forward-only — model weights only, no optimizer state, no training-rig contact.
- The probe sequence can exceed the model's `max_seq_len`, so the forward is run in windows of `max_seq_len` and the captured `h` is concatenated (faithful, since causal `h` for token i depends only on tokens <= i).
- Built-in sanity checks: CE from raw vs centered logits must match (`sanity_CE_abs_diff` ~0, confirming CE-invariance) and `logZ_c` must equal `logZ - h.mu` elementwise (`decomp_err` ~0).
- VERDICT logic (priority order): GEOMETRY FIRST — `effective_rank_c` is the sharp collapse detector (healthy KEEL ~8-14; thresholds `EFF_RANK_C_WARN=7.0`, `EFF_RANK_C_CRIT=6.0`, corroborated by `spectral_concentration_c > 0.45`). A collapse is a geometry failure → investigate/rollback recent interventions (head-WD, LR, optimizer-state), and NEVER engage z-loss (it moves scale, not rank). Then the scale axis: `logZ_c` mean is classified against `KEEL_LOGZC_FAMILY_BAND = (60, 130)` — above = anomalous centered-margin inflation to investigate, below = margin not yet developed or actively compressed, in-band = normal KEEL margin → gauge subtraction only.
- Also reports an "excess" form `excess = logZ_c - log V = KL(U||p) >= 0` (Rook #144) as the family-comparable, V-independent concentration control variable, plus per-target-id quantile `freq_buckets` of `logZ_c`.
- Repo wiring mirrors `generate_neo.py`: imports `logger` and `neo_common` from `../common_fsdp2` (and `../saved_code`); uses `nc.load_model_and_tokenizer` (auto-detects KEEL) and `nc.detect_device`. Needs `numpy`, `torch`, `pyyaml`; the full path benefits from CUDA/bf16 autocast. Full data path materializes an `[Nv, V]` fp32 logits tensor, so VRAM/RAM scales with vocab size × valid tokens — the `--weights-only` path avoids this entirely. Related memory: `zloss_rowcenter_finding.md`, `zloss_rowcenter_probe_config.md`, `row_center_head.md`.
