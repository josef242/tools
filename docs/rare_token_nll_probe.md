# rare_token_nll_probe.py
> Per-checkpoint NLL audit on a fixed held-out panel, bucketed by on-panel token frequency, to check whether z-loss over-compression damaged the rare-token tail.

## What it does
This is the z-loss tail-safety audit ("Item A"). The concern: a brief z-loss
over-compression on dn2 (logZ_c dropping ~108->83 across steps 18000->18500)
flattens the centered output distribution, and if that hurts anything it should
show up first in rare-token discrimination rather than in average CE. So instead
of inferring damage from the logZ_c scalar, this probe measures the tail
directly. It evaluates a checkpoint on a fixed deterministic panel (~100-200k
tokens), buckets target tokens by their frequency on that panel (frequency
quintiles Q0=high...Q4=low plus an `ultra`-rare bucket), and reports per-bucket
NLL / target-prob / target-rank / entropy / logZ_c. The headline metric
`dNLL_bucket = NLL_bucket(ckpt) - NLL_bucket(baseline)` is computed externally by
diffing the per-checkpoint JSON outputs.

Part of the z-loss family of probes. It reuses the row-center probe's loading and
capture machinery verbatim (imports `resolve_ckpt`, `_resolve_own_groups`,
`capture_final_h` from `zloss_row_center_probe`).

## Usage
```bash
# Single-GPU run (local 4080, fla-infer env), one checkpoint -> one JSON
python rare_token_nll_probe.py --ckpt <pt> --config <config_yaml> \
    --ntokens 150000 --out rare_nll_<step>.json

# Same panel for every checkpoint: keep --seed (and --ntokens/--groups) IDENTICAL
python rare_token_nll_probe.py --ckpt <ckpt_18500> --config <yaml> \
    --seed 0 --out rare_nll_18500.json

# Large model split across multiple GPUs (model won't fit + logits chunk on one card)
python rare_token_nll_probe.py --ckpt <pt> --config <yaml> --shard balanced \
    --out rare_nll.json
```
Run the same invocation per checkpoint (baseline + candidates), then diff the
JSONs to get `dNLL_bucket`.

## Key arguments
| flag | default | meaning |
| --- | --- | --- |
| `--ckpt` | (required) | checkpoint to probe; resolved via `resolve_ckpt` |
| `--ntokens` | `150000` | panel size in tokens; keep constant across checkpoints |
| `--config` | `None` | ground-truth `config_*.yaml` used to resolve the panel groups |
| `--groups` | `None` | comma-separated group override (instead of resolving from config) |
| `--seed` | `0` | panel seed; MUST be identical across checkpoints so panels match |
| `--out` | `None` | output JSON path; if omitted, results are logged but not written |
| `--shard` | `none` | model shard strategy: `none` (single device) or `balanced` (split across all `CUDA_VISIBLE_DEVICES` via accelerate) |
| `--tok-chunk` | `1024` | per-token-metrics chunk size; lower if the metrics step OOMs (~128MB per `[chunk,V]` logits tensor at 1024) |
| `--metrics-device` | `None` | device for the metrics matmul, e.g. `cuda:3`; under `--shard` defaults to the highest visible CUDA index (shard-free), since the model packs the low cards full |

## Notes
- Comparability hinges on `(seed, groups, ntokens)` being identical across
  checkpoints — the panel is assembled deterministically (round-robin contiguous
  4096-token blocks from each group's `*_val_*.npy` shards) so the exact token
  sequence repeats, making `dNLL` strictly comparable.
- Bucketing: frequency quintiles are equal-MASS over valid tokens (each Q has
  ~equal token count), labeled so Q0=most frequent ... Q4=least frequent; the
  `ultra` bucket is the rarest 2% of unique vocab ids actually present on the
  panel (`ultra_rare_frac=0.02`, not exposed as a flag).
- `nll_mean` over all valid tokens equals plain CE; per-token logZ_c is computed
  as `logsumexp(logits - h·mu)` where `mu` is the head weight column mean (the
  centered / gauge-removed log-partition).
- Under balanced sharding the head shard may be packed full and OOM on the
  `[c,V]` logits; the probe runs the metrics matmul on a separate free card
  (highest CUDA index by default, overridable with `--metrics-device`). Single
  GPU / CPU falls back to the head weight's own device.
- Depends on `common_fsdp2` (`neo_common`, `logger`) being importable (added to
  `sys.path` relative to this file) and on `zloss_row_center_probe.py` living
  alongside it. Writes a run log to `./logs/rare_token_nll_log.txt`.
- Reads `data_root_path` from the checkpoint config (falling back to a default
  relative path); pad tokens (`cfg.pad_id`, default 0) are excluded from all
  metrics.
