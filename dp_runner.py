"""dp_runner.py — task-agnostic data-parallel checkpoint-eval runner.

The resource layer below the sweep logic in generate_neo.py: GPU inventory,
worker-group resolution, and subprocess worker management. Knows nothing about
WHAT is being evaluated — sweeps decide how to split work (example shards for
scoring tasks, whole checkpoints for generation tasks) and how to merge
results; this module owns how workers get GPUs and processes.

Worker model: each group of physical GPU ids becomes one worker slot. A child
process pinned to a group via CUDA_VISIBLE_DEVICES sees its GPUs renumbered
from 0 and should load unsharded when it sees one GPU, balanced-sharded when
it sees several (biggest GPU is listed first in the group because the balanced
loader fills visible device 0 up to its cap and spills the remainder).

Scheduling: single-GPU groups run the full model at full speed; multi-GPU
groups pipeline layers one GPU at a time (~3-5x slower measured on rig-31).
WorkerPool therefore assigns queued jobs to fast (single-GPU) groups freely,
and lets a slow (multi-GPU) group take a job only when the queue is deep
enough that waiting for a fast group would be slower (see SLOW_SPEED_RATIO).
"""

import os
import re
import subprocess
import sys
import threading
import time

# Measured on rig-31 (7B bf16): single-4080 generation ~5x a 2-GPU pipeline
# pair, and ~3x for batched scoring. Used only as a scheduling heuristic.
SLOW_SPEED_RATIO = 5

LOGGER_PORT_BASE = 29601  # NEO_LOGGER_PORT per worker: BASE + worker_idx


def gpu_inventory():
    """[(physical_idx, total_mem_gb)] for all visible CUDA devices."""
    import torch
    return [(i, torch.cuda.get_device_properties(i).total_memory / 1024**3)
            for i in range(torch.cuda.device_count())]


def group_is_feasible(group_mems, need_gb):
    """Can this list of per-GPU capacities host one replica of a need_gb model?

    Mirrors neo_common's balanced-shard behavior exactly: a single GPU loads
    unsharded (whole model + activation/context headroom); a multi-GPU group is
    filled sequentially, biggest first, under a uniform forced cap of
    int(need_gb / n * 1.5) GiB per GPU.
    """
    if len(group_mems) == 1:
        return need_gb <= group_mems[0] - 2.0
    cap = int(need_gb / len(group_mems) * 1.5)
    rem = need_gb
    for m in group_mems:
        take = min(cap, rem)
        if take > m - 0.8:  # CUDA context + activations headroom
            return False
        rem -= take
    return rem <= 0.01


def resolve_groups(spec, need_gb):
    """Resolve a --dp_groups spec into a list of physical-GPU-id lists.

    Explicit form: "0;5,1;6,2" — ';' between groups, ',' within, biggest GPU
    first within each group. 'auto': greedily pack GPUs (largest first) into
    the smallest feasible groups for a need_gb model.
    """
    if spec != 'auto':
        return [[int(x) for x in g.split(',') if x.strip()]
                for g in spec.split(';') if g.strip()]

    pool = sorted(gpu_inventory(), key=lambda x: -x[1])
    groups = []
    while pool:
        group = [pool.pop(0)]
        while not group_is_feasible([m for _, m in group], need_gb) and pool:
            group.append(pool.pop(0))
        if group_is_feasible([m for _, m in group], need_gb):
            groups.append([i for i, _ in group])
        else:
            break  # remaining GPUs can't host another replica
    return groups


def model_need_gb(checkpoint_path, half_precision=True):
    """Loaded-size estimate from the on-disk checkpoint (fp32 on disk)."""
    need = os.path.getsize(checkpoint_path) / 1024**3
    return need / 2 if half_precision else need


def child_env(group, worker_idx):
    """Environment for a worker child pinned to `group`."""
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in group)
    env['NEO_LOGGER_PORT'] = str(LOGGER_PORT_BASE + worker_idx)
    return env


def child_shard_strategy():
    """For use INSIDE a worker child: how to load for the GPUs we can see."""
    import torch
    return 'balanced' if torch.cuda.device_count() > 1 else 'none'


class WorkerResult:
    __slots__ = ('job', 'ok', 'returncode', 'captures', 'worker_idx')

    def __init__(self, job, ok, returncode, captures, worker_idx):
        self.job = job
        self.ok = ok
        self.returncode = returncode
        self.captures = captures  # list of re.Match objects from capture_re
        self.worker_idx = worker_idx


class WorkerPool:
    """Run independent jobs across GPU-group worker slots.

    make_cmd(job, worker_idx, group) -> argv list for the child process.
    capture_re: lines matching it are collected (not echoed); all other child
    output is echoed to stdout prefixed "[w<idx>] ".

    Jobs may set job['group_idx'] to pin a specific worker slot (used by
    gang-style callers); unpinned jobs go to any free slot, subject to the
    slow-group admission rule: a multi-GPU group only takes an unpinned job
    when more than SLOW_SPEED_RATIO jobs remain per free fast group, so slow
    workers help on deep queues and stay out of the way on shallow ones.

    CONTRACT: a child signals success by printing at least one line matching
    capture_re before exiting 0. Exit 0 with no capture line is treated as
    failure (a child that silently produced nothing is indistinguishable from
    one that lost its result). Set capture_required=False for children whose
    exit code alone is authoritative.
    """

    def __init__(self, groups, make_cmd, capture_re, cwd=None, log=print,
                 slow_speed_ratio=SLOW_SPEED_RATIO, capture_required=True):
        # slow_speed_ratio: how many times slower a multi-GPU pipeline group is
        # than a single-GPU group FOR THIS TASK. Batched scoring is
        # compute-bound (~3-5x); single-stream generation is per-token-
        # overhead-bound (~1x) — pass 1 to treat all groups as equal.
        self.groups = groups
        self.make_cmd = make_cmd
        self.capture_re = re.compile(capture_re)
        self.cwd = cwd or os.getcwd()
        self.log = log
        self.slow_speed_ratio = slow_speed_ratio
        self.capture_required = capture_required

    def _admit(self, worker_idx, queue_len, fast_free):
        if len(self.groups[worker_idx]) == 1:
            return True
        if self.slow_speed_ratio <= 1:
            return True  # task runs equally fast on pipeline groups
        if fast_free == 0 and all(len(g) > 1 for g in self.groups):
            return True  # no fast groups exist at all
        return queue_len > self.slow_speed_ratio * max(1, fast_free)

    def run(self, jobs):
        """Execute all jobs; returns [WorkerResult] in job order."""
        results = [None] * len(jobs)
        pending = list(range(len(jobs)))
        running = {}  # worker_idx -> (job_idx, proc, thread, captures)
        lock = threading.Lock()

        def _pump(widx, proc, captures):
            for line in proc.stdout:
                line = line.rstrip('\n')
                m = self.capture_re.match(line)
                if m:
                    captures.append(m)
                elif line.strip():
                    print(f"[w{widx}] {line}", flush=True)

        while pending or running:
            # reap finished workers
            for widx in list(running):
                job_idx, proc, thread, captures = running[widx]
                if proc.poll() is not None:
                    thread.join(timeout=10)
                    ok = proc.returncode == 0 and (
                        captures or not self.capture_required)
                    results[job_idx] = WorkerResult(
                        jobs[job_idx], ok, proc.returncode, captures, widx)
                    if not ok:
                        why = (f"exit code {proc.returncode}"
                               if proc.returncode != 0 else
                               f"exited 0 but printed no line matching "
                               f"{self.capture_re.pattern!r} (see CONTRACT in "
                               f"WorkerPool docstring)")
                        self.log(f"Worker {widx} job {job_idx} failed: {why}")
                    del running[widx]

            # schedule
            free = [w for w in range(len(self.groups)) if w not in running]
            fast_free = sum(1 for w in free if len(self.groups[w]) == 1)
            launched = False
            for widx in free:
                if not pending:
                    break
                # pinned jobs first
                pin = next((j for j in pending
                            if jobs[j].get('group_idx') == widx), None)
                if pin is None:
                    unpinned = [j for j in pending
                                if jobs[j].get('group_idx') is None]
                    if not unpinned or not self._admit(widx, len(unpinned),
                                                       fast_free):
                        continue
                    pin = unpinned[0]
                pending.remove(pin)
                cmd = self.make_cmd(jobs[pin], widx, self.groups[widx])
                captures = []
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, env=child_env(self.groups[widx], widx),
                    cwd=self.cwd)
                t = threading.Thread(target=_pump, args=(widx, proc, captures),
                                     daemon=True)
                t.start()
                running[widx] = (pin, proc, t, captures)
                launched = True

            if not launched:
                time.sleep(0.5)

        return results
