"""WD-gradient-waste probe — Part B (Nexus #167/#169): the cos(g_loss, W) test.

Part A showed every body matrix's ||W|| is RAMPING (not at equilibrium) and that
WD is ~99.8% of weight motion. The open question: is that ramp LOSS-NULL (the
RMSNorm divides the scale out -> cosmetic inflation, survivable) or LOSS-COUPLED
(the loss is being dragged by the growth -> real distortion)?

Decisive metric, per weight matrix:
    cos(g_loss, W) = <g_loss, W> / (||g_loss|| ||W||)
where g_loss is the CE-ONLY gradient (no WD, no z-loss, no aux weighting).
  - For a perfectly scale-invariant (pre-norm) matrix, L(cW)=L(W) =>
    <g_loss, W> = 0 => cos = 0. WD's pull (-lambda*W, purely radial / parallel
    to W) is then ENTIRELY in the loss-null direction => wasted but harmless.
  - cos meaningfully != 0 => the loss DOES have a radial component => WD is
    fighting real signal / the ramp is loss-coupled => actionable.

wasted_wd_frac = ||component of W orthogonal to g_loss|| / ||W|| = sqrt(1 - cos^2)
  (since lambda*W is parallel to W, the WD vector's wasted fraction == this).

Read-only: forward + backward on a few real batches, read p.grad. NO optimizer
step, NO weight mutation, NO z-loss/aux folding. Reuses the row-center probe's
loader + panel builder. Per matrix also reports ||W||, ||g_loss||, and the WD/
loss magnitude ratio lambda*||W|| / ||g_loss|| (how much WD outweighs the signal).

Run (GPU; 4080 single-device or rig --shard balanced):
    python wd_waste_partb.py --ckpt <pt> --config <yaml> [--nbatch 4] [--wd 0.02] \
        [--shard balanced] [--out partb_<step>.json]
"""
import os
import re
import sys
import json
import time
import argparse

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("../common_fsdp2", "../saved_code"):
    _ap = os.path.normpath(os.path.join(_HERE, _p))
    if _ap not in sys.path:
        sys.path.insert(0, _ap)
import logger  # noqa: E402
logger._instance.set_logdir("./logs")
logger._instance.set_default_logfile("wd_waste_partb_log.txt")
logger._instance.set_rank(0)
import neo_common as nc  # noqa: E402
from zloss_row_center_probe import resolve_ckpt, _resolve_own_groups  # noqa: E402
from rare_token_nll_probe import build_panel  # noqa: E402


def _classify(name):
    """Tag a param by its norm-relationship (scale-invariance class), for grouping
    the cos results. Pre-norm body matrices are the ones expected to be ~loss-null
    in the scale direction; the head is the common-mode (not scale) case."""
    if name.startswith('output.'):
        return 'head'
    if 'attention.wo' in name or 'feed_forward.w2' in name:
        return 'body_proj_prenorm'      # output feeds Post-LN directly — cleanest scale-invariance
    if any(s in name for s in ('attention.wq', 'attention.wk', 'attention.wv',
                               'feed_forward.w1', 'feed_forward.w3')):
        return 'body_in_prenorm'        # reads normed input; output enters attn/ffn then Post-LN
    if 'tok_embeddings' in name:
        return 'embedding'
    return 'other'


def run(ckpt, config_path, nbatch=4, seq=2048, wd=0.02, shard_strategy="none",
        groups_override=None, out_path=None, seed=0):
    device = nc.detect_device(None)
    path = resolve_ckpt(ckpt)
    step = int(re.search(r"_(\d+)\.pt", os.path.basename(path)).group(1)) \
        if re.search(r"_(\d+)\.pt", os.path.basename(path)) else None
    logger.print_and_log(f"=== WD-waste Part B (cos g_loss,W): {os.path.basename(path)} "
                         f"on {device} (shard={shard_strategy}) ===")

    t0 = time.time()
    model, enc, cfg = nc.load_model_and_tokenizer(
        path, device=device, half_precision=True, shard_strategy=shard_strategy, use_keel=None,
    )
    model.train()  # enable grad path (activation ckpt etc.); we won't step the optimizer
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

    # Accumulate CE-only gradients over a few batches (no z-loss, no aux weighting).
    # We call the model to get its main CE loss and backward THAT only.
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    for p in raw.parameters():
        p.grad = None
    msl = getattr(getattr(model, "params", None), "max_seq_len", None) or seq
    win = min(seq, int(msl))
    total_tok = win * nbatch
    tokens, targets = build_panel(data_root, groups, total_tok, device, seed=seed)
    logger.print_and_log(f"panel: {tuple(tokens.shape)} window={win} nbatch~{nbatch}")

    n_done = 0
    for s in range(0, tokens.size(1) - 1, win):
        x = tokens[:, s:s + win]
        y = targets[s:s + win]
        if x.size(1) < 8:
            break
        out = model(x)
        # model(x) returns logits OR (logits,...); compute plain CE ourselves so we
        # KNOW it's CE-only (no z-loss/aux). Flatten [1,T,V] vs [T].
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = logits.reshape(-1, logits.size(-1)).float()
        loss = torch.nn.functional.cross_entropy(
            logits, y.reshape(-1), ignore_index=pad_id)
        (loss / nbatch).backward()
        n_done += 1
        if n_done >= nbatch:
            break
    logger.print_and_log(f"accumulated CE-only grads over {n_done} batch(es)")

    # Per-matrix cos(g_loss, W). Operate on local shards (DTensor) but the cosine
    # is a global inner-product ratio -> reduce <g,W>, ||g||^2, ||W||^2 over the
    # param's mesh. For non-DTensor (single-device) it's just local.
    import torch.distributed as dist
    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        DTensor = ()

    rows = []
    for name, p in raw.named_parameters():
        if p.grad is None or p.dim() != 2:
            continue
        is_dt = isinstance(p, DTensor)
        W = (p._local_tensor if is_dt else p).detach().float()
        G = (p.grad._local_tensor if isinstance(p.grad, DTensor) else p.grad).detach().float()
        dot = (W * G).sum()
        wsq = (W * W).sum()
        gsq = (G * G).sum()
        if is_dt and dist.is_available() and dist.is_initialized():
            t = torch.stack([dot, wsq, gsq])
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=p.device_mesh.get_group())
            dot, wsq, gsq = t[0], t[1], t[2]
        wn = wsq.clamp_min(0).sqrt().item()
        gn = gsq.clamp_min(0).sqrt().item()
        cos = (dot.item() / (wn * gn)) if (wn > 0 and gn > 0) else 0.0
        cos = max(-1.0, min(1.0, cos))
        rows.append({
            'name': name, 'cls': _classify(name),
            'w_norm': wn, 'g_loss_norm': gn,
            'cos_gW': cos,
            'wasted_wd_frac': (1.0 - cos * cos) ** 0.5,
            'wd_over_loss': (wd * wn / gn) if gn > 0 else float('inf'),
        })

    # ---- report ----
    def _agg(cls):
        sub = [r for r in rows if r['cls'] == cls]
        if not sub:
            return None
        import statistics as st
        coss = [r['cos_gW'] for r in sub]
        return {
            'n': len(sub),
            'cos_mean': sum(coss) / len(sub),
            'cos_absmean': sum(abs(c) for c in coss) / len(sub),
            'cos_max_abs': max(abs(c) for c in coss),
            'cos_median': st.median(coss),
            'wd_over_loss_median': st.median([r['wd_over_loss'] for r in sub if r['wd_over_loss'] != float('inf')] or [0]),
        }

    logger.print_and_log("\n=== cos(g_loss, W) by class (|cos|~0 => loss-null/survivable; "
                         "|cos| meaningfully >0 => loss-coupled/harmful) ===")
    summary = {}
    for cls in ('body_proj_prenorm', 'body_in_prenorm', 'embedding', 'head', 'other'):
        a = _agg(cls)
        summary[cls] = a
        if a:
            logger.print_and_log(
                f"  {cls:18}: n={a['n']:3d}  cos|mean|={a['cos_absmean']:.4f}  "
                f"cos_median={a['cos_median']:+.4f}  max|cos|={a['cos_max_abs']:.4f}  "
                f"wd/loss(med)={a['wd_over_loss_median']:.1f}x")

    # Most loss-coupled individual matrices (largest |cos|)
    logger.print_and_log("\nTop-10 matrices by |cos(g_loss,W)| (most loss-coupled):")
    for r in sorted(rows, key=lambda r: abs(r['cos_gW']), reverse=True)[:10]:
        logger.print_and_log(
            f"  {r['name']:42} cos={r['cos_gW']:+.4f} wasted={r['wasted_wd_frac']:.4f} "
            f"||W||={r['w_norm']:7.1f} wd/loss={r['wd_over_loss']:.1f}x")

    result = {'checkpoint': os.path.basename(path), 'step': step, 'wd': wd,
              'n_batches': n_done, 'summary': summary, 'per_matrix': rows}
    if out_path:
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=1)
        logger.print_and_log(f"\nwrote {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--nbatch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--shard", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    go = [g.strip() for g in a.groups.split(",")] if a.groups else None
    run(a.ckpt, a.config, nbatch=a.nbatch, seq=a.seq, wd=a.wd, shard_strategy=a.shard,
        groups_override=go, out_path=a.out, seed=a.seed)


if __name__ == "__main__":
    main()
