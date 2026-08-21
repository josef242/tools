# dp_runner.py
> Task-agnostic data-parallel checkpoint-eval runner — the resource layer under generate_neo's `--dp_groups` sweeps: GPU inventory/grouping, worker-subprocess management, and speed-aware job scheduling.

## What it does
Sits below the sweep logic: sweeps decide *what* to evaluate and how to split/merge work; this module decides *where* it runs. It resolves GPU worker groups (explicit `0;5,1;6,2` spec or `auto` packing from checkpoint size + per-GPU VRAM, simulating `neo_common`'s balanced-shard fill so every emitted group is loadable), pins each worker child via `CUDA_VISIBLE_DEVICES` (+ a distinct `NEO_LOGGER_PORT`), streams child output with `[w#]` prefixes, captures machine-readable result lines, and isolates failures per job.

## Key pieces
- `resolve_groups(spec, need_gb)` / `model_need_gb(ckpt_path)` — group resolution; `auto` greedily packs largest-first with a feasibility simulation mirroring the loader exactly (single GPU: whole model + 2 GB headroom; multi-GPU: uniform `int(need/n*1.5)` GiB cap filled biggest-first).
- `child_shard_strategy()` — for use inside a child: `balanced` if it sees >1 GPU, else `none`.
- `WorkerPool(groups, make_cmd, capture_re, slow_speed_ratio, capture_required)` — runs jobs across group slots. Jobs may pin a slot (`group_idx`, gang-style — hella) or float (coherence). `slow_speed_ratio` encodes how much slower a multi-GPU pipeline group is *for this task*: ~3–5 for batched scoring (compute-bound), 1 for single-stream generation (per-token-overhead-bound; a 2-GPU pair measured ~4.8 tok/s vs a solo 4080's ~5.2 on a 7B). Slow groups only take floating jobs when the queue is deep enough that they help.

## Contract
A child signals success by printing ≥1 line matching `capture_re` before exiting 0; exit-0-with-no-capture is a failure (set `capture_required=False` to trust exit codes alone). Children are spawned with cwd = the tools directory.

## Notes
- Group order matters: biggest GPU first, because the balanced loader fills visible device 0 to its cap and spills the remainder.
- Concurrent model loads: host RAM must hold (workers × on-disk checkpoint size) transiently.
- Clients today: `generate_neo.py` `--hella_sweep --dp_groups` (example shards, pinned gang, weighted 3:1) and `--coherence_sweep --dp_groups` (whole-checkpoint jobs, floating, ratio 1). The milestone planning they share lives in `generate_neo._plan_milestone_steps`.
