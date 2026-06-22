"""Finite-rescale invariance probe (Chatty review, WD_WASTE_ANALYSIS.md).

DECISION-MAKER for body renorm: cos(g_loss,W)~0 proves the INFINITESIMAL radial
direction is loss-null, but NOT that L(cW)=L(W) for a large finite c. This probe
tests the finite question directly: rescale each weight CLASS by c and measure the
forward-output change.

Per class in {wo, w2, wq, wk, wv, w1, w3} (and the head, as a sanity control), for
c in {0.95, 0.9, 0.8, 0.7}: temporarily multiply that class's matrices by c,
re-run the SAME fixed batch, measure vs the c=1 baseline:
  dCE            : |CE(c) - CE(1)|        (the headline; ~0 => finite-invariant)
  dCE_rel        : dCE / CE(1)
  dlogp_y_mean   : mean |logp(target) change|  (per-token NLL shift)
  dlogits_max    : max |logit change|     (raw output drift)
  dhidden_rms    : |final-norm-h RMS change| / baseline   (does the residual stream move?)

If CE is allclose at c=0.9 but not c=0.7 => renorm is a gentle-control tool with a
known safe range. If even 0.95x moves CE => renorm is off the table. body_proj
(wo,w2, output->Post-LN) expected safest; body_in (esp gated w1/w3) more cautious.

FORWARD-ONLY: no backward, no optimizer, no grad. Rescale is applied in-place under
no_grad and RESTORED after each measurement (W *= c, then W /= c) so the model is
unchanged at exit. Cheap -> fits a single 16GB card for the small models; --shard
balanced for the big ones (forward-only, so far lighter than Part B's backward).

Run:
    python finite_rescale_probe.py --ckpt <pt> --groups "<g1,g2>" [--shard balanced] \
        [--paired] [--out fr_<tag>.json]
  --paired: also test scaling w1 AND w3 together (the gated-MLP branch) since
            scaling one leg alone is not the same as scaling the branch.
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
logger._instance.set_default_logfile("finite_rescale_log.txt")
logger._instance.set_rank(0)
import neo_common as nc  # noqa: E402
from zloss_row_center_probe import resolve_ckpt, _resolve_own_groups, capture_final_h  # noqa: E402
from rare_token_nll_probe import build_panel  # noqa: E402

# class -> substring matcher on param name
CLASS_MATCH = {
    'wo': lambda n: n.endswith('attention.wo.weight'),
    'w2': lambda n: n.endswith('feed_forward.w2.weight'),
    'wq': lambda n: n.endswith('attention.wq.weight'),
    'wk': lambda n: n.endswith('attention.wk.weight'),
    'wv': lambda n: n.endswith('attention.wv.weight'),
    'w1': lambda n: n.endswith('feed_forward.w1.weight'),
    'w3': lambda n: n.endswith('feed_forward.w3.weight'),
    'head': lambda n: n == 'output.weight',
    # paired gated-MLP branch (w1 AND w3 together)
    'w1w3': lambda n: n.endswith('feed_forward.w1.weight') or n.endswith('feed_forward.w3.weight'),
}


def _local(p):
    return p._local_tensor if hasattr(p, '_local_tensor') else p


def _forward_ce_h(model, tokens, targets, pad_id, msl):
    """Forward the fixed batch; return (CE, logp_target[Nv], logits_flat[Nv,V] on cpu,
    hidden_rms). Uses the eval-branch (no targets) so we get logits, computes CE
    ourselves. Windowed by max_seq_len. No grad."""
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    win = int(msl) if msl else tokens.size(1)
    win = max(1, min(win, tokens.size(1)))
    # capture hidden via the existing helper (windowed) for hidden-RMS
    h = capture_final_h(model, tokens, max_seq_len=win)   # [N, D] fp32, no_grad
    hidden_rms = h.pow(2).mean().sqrt().item()
    # logits per window (eval branch returns (logits, None))
    ce_num = 0.0; ce_den = 0
    logp_chunks = []; logit_max_ref = []
    with torch.no_grad():
        for s in range(0, tokens.size(1), win):
            x = tokens[:, s:s+win]
            out = model(x)
            logits = (out[0] if isinstance(out, (tuple, list)) else out)
            logits = logits.reshape(-1, logits.size(-1)).float()
            y = targets[s:s+win].reshape(-1)
            valid = y != pad_id
            lv = logits[valid]; yv = y[valid]
            lse = torch.logsumexp(lv, dim=-1)
            tgt_logit = lv.gather(1, yv.unsqueeze(1)).squeeze(1)
            logp = tgt_logit - lse
            ce_num += (-(logp)).sum().item(); ce_den += int(valid.sum())
            logp_chunks.append(logp.cpu())
    ce = ce_num / max(1, ce_den)
    return ce, torch.cat(logp_chunks), hidden_rms


@torch.no_grad()
def run(ckpt, groups_override=None, config_path=None, cs=(0.95, 0.9, 0.8, 0.7),
        ntokens=4096, classes=None, paired=False, shard_strategy="none",
        out_path=None, seed=0):
    device = nc.detect_device(None)
    path = resolve_ckpt(ckpt)
    step = int(re.search(r"_(\d+)\.pt", os.path.basename(path)).group(1)) \
        if re.search(r"_(\d+)\.pt", os.path.basename(path)) else None
    logger.print_and_log(f"=== finite-rescale probe: {os.path.basename(path)} on {device} ===")
    t0 = time.time()
    model, enc, cfg = nc.load_model_and_tokenizer(
        path, device=device, half_precision=True, shard_strategy=shard_strategy, use_keel=None)
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

    msl = getattr(getattr(model, "params", None), "max_seq_len", None)
    tokens, targets = build_panel(data_root, groups, ntokens, device, seed=seed)
    logger.print_and_log(f"panel: {tuple(tokens.shape)}")

    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    named = dict(raw.named_parameters())

    # baseline
    ce0, logp0, hrms0 = _forward_ce_h(model, tokens, targets, pad_id, msl)
    logger.print_and_log(f"baseline CE={ce0:.6f}  hidden_rms={hrms0:.4f}")

    if classes is None:
        classes = ['wo', 'w2', 'wq', 'wk', 'wv', 'w1', 'w3', 'head']
        if paired:
            classes.append('w1w3')

    result = {'checkpoint': os.path.basename(path), 'step': step,
              'baseline': {'CE': ce0, 'hidden_rms': hrms0}, 'classes': {}}

    for cls in classes:
        match = CLASS_MATCH[cls]
        params = [named[n] for n in named if match(n)]
        if not params:
            continue
        rows = []
        for c in cs:
            # rescale in place (local shard), measure, restore
            for p in params:
                _local(p).mul_(c)
            ce, logp, hrms = _forward_ce_h(model, tokens, targets, pad_id, msl)
            for p in params:
                _local(p).div_(c)   # restore
            dlogp = (logp - logp0).abs()
            rows.append({
                'c': c,
                'dCE': abs(ce - ce0),
                'dCE_rel': abs(ce - ce0) / max(1e-9, ce0),
                'dlogp_y_mean': dlogp.mean().item(),
                'dlogp_y_p99': torch.quantile(dlogp, 0.99).item(),
                'dhidden_rms_rel': abs(hrms - hrms0) / max(1e-9, hrms0),
                'CE': ce,
            })
        result['classes'][cls] = {'n_matrices': len(params), 'rescales': rows}
        # compact log line per class
        cells = " ".join(f"c{r['c']}:dCE={r['dCE']:.2e}" for r in rows)
        logger.print_and_log(f"  {cls:6} (n={len(params):3d}): {cells}")

    # verdict: largest c with dCE_rel < 1e-3 per class (the "safe range")
    logger.print_and_log("\n=== SAFE RANGE (largest |1-c| with dCE_rel < 1e-3) ===")
    for cls, d in result['classes'].items():
        safe = [r['c'] for r in d['rescales'] if r['dCE_rel'] < 1e-3]
        worst_safe = min(safe) if safe else None   # smallest c (largest deviation) still safe
        tag = f"safe down to c={worst_safe}" if worst_safe is not None else "NOT safe even at c=0.95"
        logger.print_and_log(f"  {cls:6}: {tag}")

    if out_path:
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=1)
        logger.print_and_log(f"wrote {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--ntokens", type=int, default=4096)
    ap.add_argument("--cs", default="0.95,0.9,0.8,0.7", help="comma-sep scale factors")
    ap.add_argument("--classes", default=None, help="comma-sep subset of classes")
    ap.add_argument("--paired", action="store_true", help="also test w1w3 together")
    ap.add_argument("--shard", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    go = [g.strip() for g in a.groups.split(",")] if a.groups else None
    cs = tuple(float(x) for x in a.cs.split(","))
    classes = [c.strip() for c in a.classes.split(",")] if a.classes else None
    run(a.ckpt, groups_override=go, config_path=a.config, cs=cs, ntokens=a.ntokens,
        classes=classes, paired=a.paired, shard_strategy=a.shard, out_path=a.out, seed=a.seed)


if __name__ == "__main__":
    main()
