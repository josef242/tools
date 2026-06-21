"""Rare-token NLL bucket probe (Item A — the z-loss tail-safety audit).

Question: did the brief z-loss over-compression (dn2 logZ_c 108->83 across
18000->18500) leave a mark on the RARE-TOKEN tail? logZ_c = log V + KL(U||p), so
lowering logZ_c flattens the centered distribution — if that hurt anything it
shows up FIRST in rare-token discrimination, NOT in average CE. So we measure the
tail directly instead of inferring it from the logZ_c scalar.

Method: evaluate each checkpoint on the SAME fixed held-out panel (~100-200k
tokens). Bucket target tokens by their frequency ON THE PANEL (quintiles Q0 high
-> Q4 low, plus one ultra-rare bucket). Per checkpoint x bucket: mean/median/p95
NLL, target-prob quantiles, target-rank quantiles, entropy of p(.|x). Also log
logZ_c. Headline: dNLL_bucket(ckpt) = NLL_bucket(ckpt) - NLL_bucket(baseline),
computed externally across the per-ckpt JSON outputs.

Reuses the row-center probe's load / --config / windowed-h-capture machinery.

Run (local 4080, fla-infer):
    python rare_token_nll_probe.py --ckpt <pt> --config <yaml> --ntokens 150000 \
        --out rare_nll_<step>.json
"""
import os
import re
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("../common_fsdp2", "../saved_code"):
    _ap = os.path.normpath(os.path.join(_HERE, _p))
    if _ap not in sys.path:
        sys.path.insert(0, _ap)
import logger  # noqa: E402
logger._instance.set_logdir("./logs")
logger._instance.set_default_logfile("rare_token_nll_log.txt")
logger._instance.set_rank(0)
import neo_common as nc  # noqa: E402

# Reuse the row-center probe's load/config/capture helpers verbatim — same model
# load, same ground-truth group resolution, same windowed h-capture.
from zloss_row_center_probe import (
    resolve_ckpt, _resolve_own_groups, capture_final_h,
)


def build_panel(data_root, groups, ntokens, device, seed=0):
    """A FIXED, deterministic panel of ~ntokens tokens, assembled round-robin
    across the groups' val shards. Same (seed, groups, ntokens) -> identical
    token sequence on every checkpoint, so dNLL is strictly comparable. Unlike
    the row-center probe's build_val_batch (one small chunk/group, early break),
    this draws contiguous blocks round-robin until it reaches ntokens, so the
    panel is large enough for rare-bucket statistics. Returns (tokens[1,N],
    targets[N])."""
    candidate = [data_root] if os.path.isabs(data_root) else [
        os.path.normpath(os.path.join(_HERE, "../mara_fsdp2", data_root)),
        os.path.abspath(data_root),
    ]
    root = next((p for p in candidate if os.path.isdir(p)), None)
    if root is None:
        raise FileNotFoundError(f"data root not found, tried {candidate}")
    import glob
    rng = np.random.default_rng(seed)
    need = ntokens + 1
    # Per round-robin visit, take a block of this many tokens from a group.
    block = 4096
    shard_lists = {}
    for g in groups:
        s = sorted(glob.glob(os.path.join(root, g, "*_val_*.npy")))
        if s:
            shard_lists[g] = s
        else:
            logger.print_and_log(f"  [warn] no val shards for '{g}', skipping")
    if not shard_lists:
        raise RuntimeError("no usable val shards for any group")
    glist = list(shard_lists.keys())
    chunks, total = [], 0
    gi = 0
    guard = 0
    while total < need and guard < 100000:
        guard += 1
        g = glist[gi % len(glist)]
        gi += 1
        shards = shard_lists[g]
        sh = shards[int(rng.integers(0, len(shards)))]
        arr = np.load(sh, mmap_mode="r")
        if arr.shape[0] < block + 1:
            continue
        start = int(rng.integers(0, arr.shape[0] - block))
        chunks.append(np.asarray(arr[start:start + block]).astype(np.int64))
        total += block
    flat = np.concatenate(chunks)[:need]
    toks = torch.from_numpy(flat).long().to(device)
    return toks[:-1].unsqueeze(0), toks[1:]


def panel_freq_buckets(targets, pad_id, nquint=5, ultra_rare_frac=0.02):
    """Assign each valid target token to a bucket by ITS frequency ON THE PANEL.
    Q0=most frequent ... Q(nquint-1)=least frequent, by equal-mass quantiles of
    the per-token count; plus an 'ultra' bucket = the rarest ultra_rare_frac of
    UNIQUE vocab ids seen. Returns dict bucket_name -> bool mask over valid
    positions, plus the per-id count table (for reporting)."""
    valid = targets != pad_id
    tv = targets[valid]
    V = int(targets.max().item()) + 1
    counts = torch.bincount(tv, minlength=V).float()        # [V] on-panel counts
    tok_count = counts[tv]                                   # [Nv] count of each token
    # Equal-MASS quintiles over tokens (so each bucket has ~equal #tokens),
    # ordered by frequency: sort valid positions by their token's panel count.
    order = torch.argsort(tok_count)                         # ascending = rare->freq
    Nv = tv.numel()
    masks = {}
    # Q0 = most frequent ... so reverse: highest-count first.
    # Build quintiles on the ascending order then label from the top.
    for q in range(nquint):
        lo = q * Nv // nquint
        hi = (q + 1) * Nv // nquint
        sel = order[lo:hi]                                   # ascending freq slice
        m = torch.zeros(Nv, dtype=torch.bool, device=targets.device)
        m[sel] = True
        # name so Q0=high-freq: ascending slice q -> quintile (nquint-1-q)
        masks[f"Q{nquint-1-q}"] = m
    # Ultra-rare: tokens whose panel count is in the bottom ultra_rare_frac of
    # UNIQUE ids present (i.e. the genuinely rare vocab, not just equal-mass tail).
    present = torch.where(counts > 0)[0]
    pc = counts[present]
    k = max(1, int(present.numel() * ultra_rare_frac))
    rare_ids = present[torch.argsort(pc)[:k]]               # k rarest unique ids
    rare_set = torch.zeros(V, dtype=torch.bool, device=targets.device)
    rare_set[rare_ids] = True
    masks["ultra"] = rare_set[tv]
    return masks, counts, valid


def _quantiles(x, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    if x.numel() == 0:
        return {f"p{int(q*100)}": None for q in qs}
    xf = x.float()
    return {f"p{int(q*100)}": torch.quantile(xf, q).item() for q in qs}


def per_token_metrics(h, W, targets, pad_id, tok_chunk=1024):
    """For each valid position compute NLL, target prob, target rank, and entropy
    of p(.|x), plus per-position logZ_c. Chunked over tokens so the [chunk, V]
    logits fit on the GPU. Memory-lean: one [c,V] logits tensor at a time, results
    accumulated on CPU. tok_chunk=1024 -> ~128MB/logits tensor.

    Device-correct under model sharding: the head (W) may live on a non-cuda:0
    shard, so we do all chunk math on W's device and move inputs there per chunk."""
    wdev = W.device
    Wf = W.float()
    V, D = Wf.shape
    mu = Wf.mean(dim=0)
    valid = (targets != pad_id)
    h_v = h[valid].to(wdev)
    tgt_v = targets[valid].to(wdev)
    Nv = h_v.shape[0]
    # Accumulate on CPU to keep GPU headroom for the [c,V] logits.
    nll = torch.empty(Nv); tprob = torch.empty(Nv); trank = torch.empty(Nv)
    ent = torch.empty(Nv); logZc = torch.empty(Nv)
    for s in range(0, Nv, tok_chunk):
        e = min(s + tok_chunk, Nv)
        hc = h_v[s:e]                                   # [c, D]
        tc = tgt_v[s:e]                                 # [c]
        logits = hc @ Wf.t()                            # [c, V]  (the only big tensor)
        logZ = torch.logsumexp(logits, dim=-1)          # [c]
        tgt_logit = logits.gather(1, tc.unsqueeze(1)).squeeze(1)
        nll[s:e] = (logZ - tgt_logit).cpu()             # = -log p(target)
        tprob[s:e] = (tgt_logit - logZ).exp().cpu()
        # rank: # logits strictly exceeding the target's (0 = argmax-correct).
        # The bool is transient; summed immediately, no [c,V] kept beyond logits.
        trank[s:e] = (logits > tgt_logit.unsqueeze(1)).sum(dim=1).float().cpu()
        # entropy = logZ - (sum_j p_j * logit_j); p = softmax. Compute the
        # weighted-logit sum via (softmax(logits) * logits).sum without holding a
        # separate logp tensor: reuse logits in place for the exp.
        sm = torch.softmax(logits, dim=-1)              # [c, V]
        ent[s:e] = (logZ - (sm * logits).sum(dim=1)).cpu()   # H(p) in nats
        del sm
        # centered logZ (h.mu is a uniform per-token shift): logsumexp(logits - h.mu)
        hdotmu = hc @ mu
        logZc[s:e] = torch.logsumexp(logits - hdotmu.unsqueeze(1), dim=-1).cpu()
        del logits
    return dict(nll=nll, tprob=tprob, trank=trank, ent=ent, logZc=logZc), tgt_v.cpu(), valid


def run(ckpt, ntokens, out_path, config_path=None, groups_override=None, seed=0,
        shard_strategy="none", tok_chunk=1024):
    device = nc.detect_device(None)
    path = resolve_ckpt(ckpt)
    step = int(re.search(r"_(\d+)\.pt", os.path.basename(path)).group(1)) \
        if re.search(r"_(\d+)\.pt", os.path.basename(path)) else None
    logger.print_and_log(f"=== rare-token NLL probe: {os.path.basename(path)} on {device} "
                         f"(shard={shard_strategy}) ===")

    t0 = time.time()
    model, enc, cfg = nc.load_model_and_tokenizer(
        path, device=device, half_precision=True, shard_strategy=shard_strategy, use_keel=None,
    )
    model.eval()
    pad_id = int(getattr(cfg, "pad_id", 0) or 0)
    logger.print_and_log(f"loaded in {time.time()-t0:.1f}s; pad_id={pad_id}")

    data_root = getattr(cfg, "data_root_path", "../../notebooks/datasets/tokenized/llama/")
    if groups_override:
        groups, gsrc = groups_override, "CLI override"
    else:
        groups, gsrc = _resolve_own_groups(cfg, path, config_path=config_path)
    if not groups:
        raise RuntimeError("could not determine groups; pass --config or --groups")
    logger.print_and_log(f"groups [{gsrc}]: {groups}")

    tokens, targets = build_panel(data_root, groups, ntokens, device, seed=seed)
    logger.print_and_log(f"panel: {tuple(tokens.shape)} (+targets {tuple(targets.shape)}) seed={seed}")
    msl = getattr(getattr(model, "params", None), "max_seq_len", None) \
        or getattr(cfg, "max_seq_len", None)
    h = capture_final_h(model, tokens, max_seq_len=msl)
    logger.print_and_log(f"captured h: {tuple(h.shape)} rms={h.pow(2).mean().sqrt().item():.4f}")

    W = model.output.weight
    pm, tgt_v, valid = per_token_metrics(h, W, targets, pad_id, tok_chunk=tok_chunk)
    # bucketing on CPU (cheap; aligns with the CPU-accumulated pm tensors)
    masks, counts, _ = panel_freq_buckets(targets.cpu(), pad_id)

    def bucket_block(mask):
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            return None
        nll = pm["nll"][idx]
        return {
            "n": int(idx.numel()),
            "nll_mean": nll.mean().item(),
            "nll_median": nll.median().item(),
            "nll_p95": torch.quantile(nll.float(), 0.95).item(),
            "tgt_prob_q": _quantiles(pm["tprob"][idx]),
            "tgt_rank_q": _quantiles(pm["trank"][idx]),
            "entropy_mean": pm["ent"][idx].mean().item(),
            "logZ_c_mean": pm["logZc"][idx].mean().item(),
            "panel_count_range": [int(counts[tgt_v[idx]].min().item()),
                                  int(counts[tgt_v[idx]].max().item())],
        }

    buckets = {name: bucket_block(m) for name, m in masks.items()}
    overall = {
        "n_valid": int(valid.sum().item()),
        "nll_mean": pm["nll"].mean().item(),            # == CE
        "nll_p95": torch.quantile(pm["nll"].float(), 0.95).item(),
        "logZ_c_mean": pm["logZc"].mean().item(),
        "entropy_mean": pm["ent"].mean().item(),
    }
    logger.print_and_log(f"[overall] {json.dumps(overall, indent=2)}")
    logger.print_and_log(f"[buckets] {json.dumps(buckets, indent=2)}")

    result = {
        "checkpoint": os.path.basename(path), "step": step,
        "ntokens": ntokens, "seed": seed, "panel_groups": groups,
        "overall": overall, "buckets": buckets,
    }
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.print_and_log(f"wrote {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ntokens", type=int, default=150000)
    ap.add_argument("--config", default=None, help="ground-truth config_*.yaml for groups")
    ap.add_argument("--groups", default=None, help="comma-separated group override")
    ap.add_argument("--seed", type=int, default=0, help="panel seed (SAME across ckpts!)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard", default="none",
                    help="model shard strategy across visible GPUs: 'none' (single "
                         "device, default) | 'balanced' (split the model across all "
                         "CUDA_VISIBLE_DEVICES via accelerate — use when one card "
                         "can't hold the 23GB model + logits chunk).")
    ap.add_argument("--tok-chunk", type=int, default=1024,
                    help="per-token-metrics chunk size (lower if OOM in the metrics step)")
    a = ap.parse_args()
    go = [g.strip() for g in a.groups.split(",")] if a.groups else None
    run(a.ckpt, a.ntokens, a.out, config_path=a.config, groups_override=go, seed=a.seed,
        shard_strategy=a.shard, tok_chunk=a.tok_chunk)


if __name__ == "__main__":
    main()
