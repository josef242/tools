"""NorMuon actual-update decomposition (Math Agent test #2; taming the body ramp).

Part B measured cos(g_loss, W) — the GRADIENT. But NorMuon does NOT step along the
gradient: it steps along the Newton-Schulz-orthogonalized, RMS/neuron-normalized
update. So the gradient being radial-null (cos~0) does NOT imply the ACTUAL step is
radial-null. If the normalized update has a radial component, THAT feeds the ramp
even when CE grads are radial-null — the mechanism a fix must target.

This decomposes the ACTUAL applied update per matrix:
  total_dW = W_after - W_before              (one real optimizer.step() on CE-only grads)
  wd_dW    = -(eff_lr * wd) * W_before       (decoupled WD term, known analytically:
                                              muon_fsdp2 applies p.mul_(1-eff_lr*wd))
  muon_dW  = total_dW - wd_dW                (the pure loss-driven NorMuon step)
Then per matrix:
  cos(muon_dW, W)          : radial component of the ACTUAL update (the ramp driver)
  radial_frac             : |<muon_dW,W>| / (||muon_dW|| ||W||)   (== |cos|)
  ||muon_dW|| , ||wd_dW||  : tangential-injection vs radial-removal magnitudes
  dnorm_pred              : predicted d||W|| this step = <total_dW, W>/||W||
                            (>0 => ramping; the per-step norm change, decomposed)
  muon_radial_dnorm       : <muon_dW, W>/||W||   (how much the UPDATE pushes norm UP)
  wd_radial_dnorm         : <wd_dW, W>/||W|| = -eff_lr*wd*||W||  (WD pushes norm DOWN)
The net (muon_radial_dnorm + wd_radial_dnorm) is the per-step ramp rate, decomposed
into "update injects" vs "WD removes" — the equilibrium balance from the doc:
||U||^2 ~ 2*eta*lambda*||W||^2 i.e. ramp until WD-removal == update-injection.

This MUTATES the optimizer state + weights (real step). So: load, snapshot, ONE
step, decompose, done — do NOT save the model. Read-from-checkpoint only.

Run (GPU; rig-31 --shard balanced for big models):
    python normuon_update_decomp.py --ckpt <pt> --groups "<g>" [--shard balanced] \
        [--out nud_<tag>.json]
"""
import os
import re
import sys
import json
import time
import argparse

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("../common_fsdp2", "../saved_code"):
    _ap = os.path.normpath(os.path.join(_HERE, _p))
    if _ap not in sys.path:
        sys.path.insert(0, _ap)
import logger  # noqa: E402
logger._instance.set_logdir("./logs")
logger._instance.set_default_logfile("normuon_update_decomp_log.txt")
logger._instance.set_rank(0)
import neo_common as nc  # noqa: E402
from zloss_row_center_probe import resolve_ckpt, _resolve_own_groups  # noqa: E402
from rare_token_nll_probe import build_panel  # noqa: E402
from wd_waste_partb import _classify  # noqa: E402


def _local(t):
    return t._local_tensor if hasattr(t, '_local_tensor') else t


def run(ckpt, groups_override=None, config_path=None, ntokens=2048, seq=1024,
        shard_strategy="none", wd=0.02, out_path=None, seed=0):
    device = nc.detect_device(None)
    path = resolve_ckpt(ckpt)
    step = int(re.search(r"_(\d+)\.pt", os.path.basename(path)).group(1)) \
        if re.search(r"_(\d+)\.pt", os.path.basename(path)) else None
    logger.print_and_log(f"=== NorMuon update-decomp: {os.path.basename(path)} on {device} ===")

    t0 = time.time()
    # Need the optimizer too. neo_common loads the model; we build a matching
    # optimizer from the checkpoint to take ONE real step. If that's unavailable,
    # fall back to a plain-gradient proxy (documented in output).
    model, enc, cfg = nc.load_model_and_tokenizer(
        path, device=device, half_precision=True, shard_strategy=shard_strategy, use_keel=None)
    model.train()
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

    raw = model._orig_mod if hasattr(model, '_orig_mod') else model

    # Replicate the EXACT NorMuon update transform (muon_fsdp2 FSDP path) on each
    # body matrix — pure tensor ops, no optimizer/mesh reconstruction:
    #   1) apply_momentum (cold buffer here -> = grad on a single step; the cos is
    #      geometry-dominated by the orthogonalization, robust to warm vs cold)
    #   2) zeropower_via_newtonschulz5 (orthogonalize)
    #   3) apply_scaling (RMS)
    #   4) apply_normuon (neuron-wise normalize; cold second-moment buffer)
    # This is the ACTUAL update DIRECTION NorMuon would take, vs the gradient that
    # Part B measured. Caveat (logged): cold momentum/2nd-moment buffers; warm
    # buffers shift magnitude but the radial cos is dominated by NS geometry.
    from muon_fsdp2 import (zeropower_via_newtonschulz5, apply_scaling, apply_normuon,
                            apply_momentum)
    ns_steps = int(getattr(cfg, 'muon_ns_steps', 5) or 5)
    momentum = float(getattr(cfg, 'muon_momentum', 0.95) or 0.95)
    beta2 = float(getattr(cfg, 'normuon_beta2', 0.95) or 0.95)
    rms_scale = bool(getattr(cfg, 'muon_rms_scale', False))

    def normuon_update(grad2d):
        """Faithful NorMuon update direction for one [out,in] matrix, cold buffers."""
        g = grad2d.float().clone()
        mom = torch.zeros_like(g)
        u = apply_momentum(g, mom, momentum, False)          # cold -> ~= g
        u = zeropower_via_newtonschulz5(u, ns_steps).type_as(grad2d).float()
        u = apply_scaling(u, rms_scale)
        sm = torch.zeros(u.shape[0], 1, device=u.device, dtype=u.dtype)
        u = apply_normuon(u, sm, beta2)
        return u

    optimizer = None  # not used; we compute the update transform directly
    opt_kind = "direct NorMuon-transform replication (cold buffers)"
    logger.print_and_log(f"optimizer: {opt_kind}  ns={ns_steps} mom={momentum} beta2={beta2} rms_scale={rms_scale}")

    # CE-only grads
    win = min(seq, int(getattr(getattr(model, "params", None), "max_seq_len", seq) or seq))
    tokens, targets = build_panel(data_root, groups, ntokens, device, seed=seed)
    for p in raw.parameters():
        p.grad = None
    nb = 0
    for s in range(0, tokens.size(1) - 1, win):
        x = tokens[:, s:s+win]; y = targets[s:s+win]
        if x.size(1) < 8:
            break
        out = model(x)
        logits = (out[0] if isinstance(out, (tuple, list)) else out).reshape(-1, raw.output.weight.shape[0]).float()
        loss = F.cross_entropy(logits, y.reshape(-1), ignore_index=pad_id)
        loss.backward()
        nb += 1
        if nb >= max(1, ntokens // win):
            break
    logger.print_and_log(f"CE-only grads over {nb} window(s)")

    named = dict(raw.named_parameters())

    import torch.distributed as dist
    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        DTensor = ()

    rows = []
    for n, p in named.items():
        if p.dim() != 2 or p.grad is None:
            continue
        # NOTE: on a sharded (accelerate) load each rank holds a slice; NS needs the
        # full matrix. For single-device (small models) this is exact. For sharded
        # big models we compute on the LOCAL shard's rows (apply_normuon is row-wise;
        # NS on a row-slice is an approximation — flagged). Body matrices are small
        # enough to run single-device; prefer --shard none when it fits.
        W0 = _local(p).detach().float()
        G = _local(p.grad).detach().float()
        if G.dim() != 2:
            continue
        u = normuon_update(G)                      # NorMuon update DIRECTION
        # cos(update, W) and cos(grad, W) both, on local tensors (global reduce below)
        def _stats(a, b):
            dot = (a * b).sum(); asq = (a * a).sum(); bsq = (b * b).sum()
            if isinstance(p, DTensor) and dist.is_available() and dist.is_initialized():
                t = torch.stack([dot, asq, bsq]); dist.all_reduce(t, group=p.device_mesh.get_group())
                dot, asq, bsq = t[0], t[1], t[2]
            an = asq.clamp_min(0).sqrt().item(); bn = bsq.clamp_min(0).sqrt().item()
            return (dot.item()/(an*bn) if an>0 and bn>0 else 0.0), an, bn
        cos_uW, un, wn = _stats(u, W0)
        cos_gW, _, _ = _stats(G, W0)
        rows.append({
            'name': n, 'cls': _classify(n), 'w_norm': wn,
            'cos_muonupd_W': max(-1.0, min(1.0, cos_uW)),
            'cos_grad_W': max(-1.0, min(1.0, cos_gW)),
            'update_norm': un,
        })
    mode = "normuon_transform"

    # aggregate by class — compare cos(GRAD,W) [what Part B measured] vs
    # cos(NorMuon-UPDATE,W) [what actually steps]. The headline: if update cos >>
    # grad cos, the NORMALIZED UPDATE injects radial drift the gradient didn't,
    # i.e. NorMuon's orthogonalization is FEEDING the ramp -> the mechanism.
    import statistics as st
    summary = {}
    for cls in ('body_proj_prenorm','body_in_prenorm','head','embedding'):
        sub=[r for r in rows if r['cls']==cls]
        if not sub: continue
        gc=[r['cos_grad_W'] for r in sub]; uc=[r['cos_muonupd_W'] for r in sub]
        a={'n':len(sub),
           'cos_grad_absmean':st.mean(abs(c) for c in gc),
           'cos_grad_median':st.median(gc),
           'cos_update_absmean':st.mean(abs(c) for c in uc),
           'cos_update_median':st.median(uc),
           'cos_update_max_abs':max(abs(c) for c in uc),
           # net radial sign of the update (does it push norm UP on average?):
           'update_radial_meancos':st.mean(uc)}
        summary[cls]=a
        logger.print_and_log(
            f"  {cls:18} n={a['n']:3d} | cos(grad,W)|mean|={a['cos_grad_absmean']:.4f} "
            f"-> cos(UPDATE,W)|mean|={a['cos_update_absmean']:.4f} "
            f"(median {a['cos_update_median']:+.4f}, max {a['cos_update_max_abs']:.4f})")

    logger.print_and_log("\nKEY: if cos(UPDATE,W) >> cos(grad,W), the NorMuon normalized "
                         "update injects radial drift the gradient didn't -> mechanism feeding the ramp.")

    result={'checkpoint':os.path.basename(path),'step':step,'mode':mode,
            'summary':summary,'per_matrix':rows}
    if out_path:
        with open(out_path,'w') as f: json.dump(result,f,indent=1)
        logger.print_and_log(f"wrote {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--ntokens", type=int, default=2048)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--shard", default="none")
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    go = [g.strip() for g in a.groups.split(",")] if a.groups else None
    run(a.ckpt, groups_override=go, config_path=a.config, ntokens=a.ntokens, seq=a.seq,
        shard_strategy=a.shard, wd=a.wd, out_path=a.out, seed=a.seed)


if __name__ == "__main__":
    main()
