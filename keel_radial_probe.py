"""KEEL radial-gradient probe (Math Agent plan; WD_WASTE_ANALYSIS.md mechanism).

Tests WHY pure CE produces a small anti-radial body gradient (⟨g,W⟩<0 ⟹ descent
grows ‖W‖). Hypothesis: body norm is an implicit BRANCH-GAIN knob in the KEEL
highway `x_{l+1}=Norm(α·x_l + F_l(Norm(x_l)))`. Three read-only probes, one load:

PROBE 1 — same-batch radial gradient, train vs eval forward path (resolve Part B
  vs in-situ). cos(g,W) per body matrix under model.eval() and model.train() on the
  SAME batch/targets. PLUS finite-difference: (L(e^δW)−L(e^−δW))/2δ ≈ ⟨g,W⟩ summed
  over body matrices — confirms the anti-radial gradient is a REAL loss derivative,
  not a backward/path artifact. (Probe B used eval-branch; in-situ used train-branch.)

PROBE 2 — branch-gain derivative (the mechanism confirmer). Forward-hook each KEEL
  block's branch module (attention, feed_forward) to scale its output by a per-block
  scalar gain g_l (requires_grad, =1). Read dL/d log g_l = g_l·dL/dg_l at g_l=1.
  If NEGATIVE across layers ⟹ "the model wants MORE branch relative to highway" ⟹
  directly explains ⟨g,W⟩<0 (body scale ≈ branch gain). The decisive mechanism test.

PROBE 3 — ε sensitivity. Re-measure body cos(g,W) with RMSNorm ε ∈ {1e-5,1e-6,1e-8,0}
  (temp-patched, restored). If the anti-radial cos vanishes as ε→0 ⟹ ε is the source;
  if it persists ⟹ branch/highway directional mixing dominates (Math Agent's bet).

Read-only: no optimizer, no weight mutation persisted (finite-diff/ε perturbations
are applied then restored). Run on a checkpoint; mf is the clean case (no aux/z/dropout).

Run:
    python keel_radial_probe.py --ckpt <pt> --groups "<g>" [--shard balanced] [--out k.json]
"""
import os
import re
import sys
import json
import math
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
logger._instance.set_default_logfile("keel_radial_log.txt")
logger._instance.set_rank(0)
import neo_common as nc  # noqa: E402
from zloss_row_center_probe import resolve_ckpt, _resolve_own_groups  # noqa: E402
from rare_token_nll_probe import build_panel  # noqa: E402

try:
    from torch.distributed.tensor import DTensor as _DT
except Exception:
    _DT = ()
import torch.distributed as _dist


def _local(t):
    return t._local_tensor if isinstance(t, _DT) else t


def _gsum(t, ref):
    if isinstance(ref, _DT) and _dist.is_available() and _dist.is_initialized():
        t = t.clone(); _dist.all_reduce(t, group=ref.device_mesh.get_group())
    return t


def _isbody(n):
    return any(n.endswith(s) for s in ('wo.weight', 'w2.weight', 'wq.weight', 'wk.weight',
                                       'wv.weight', 'w1.weight', 'w3.weight'))


def _cls(n):
    if n.endswith('wo.weight') or n.endswith('w2.weight'):
        return 'body_proj'
    return 'body_in'


def _ce_loss(model, x, y, pad_id, raw):
    """Forward + CE. Uses the model's eval-branch logits (returns (logits,None) when
    no targets) and computes CE externally — consistent across train/eval mode."""
    out = model(x)
    logits = (out[0] if isinstance(out, (tuple, list)) else out).reshape(-1, raw.output.weight.shape[0]).float()
    return F.cross_entropy(logits, y.reshape(-1), ignore_index=pad_id)


def _body_grad_cos(model, x, y, pad_id, raw):
    """Backward CE, return {name: cos(g,W)} for body matrices (signed)."""
    for p in raw.parameters():
        p.grad = None
    loss = _ce_loss(model, x, y, pad_id, raw)
    loss.backward()
    out = {}
    for n, p in raw.named_parameters():
        if not _isbody(n) or p.grad is None:
            continue
        W = _local(p).detach().float(); G = _local(p.grad).detach().float()
        dot = _gsum((W*G).sum(), p).item()
        wn = _gsum((W*W).sum(), p).clamp_min(0).sqrt().item()
        gn = _gsum((G*G).sum(), p).clamp_min(0).sqrt().item()
        out[n] = (dot/(wn*gn)) if (wn > 0 and gn > 0) else 0.0
    return out, loss.item()


@torch.no_grad()
def _finite_diff_radial(model, x, y, pad_id, raw, delta=1e-3):
    """Sum over body matrices of d/dlog||W|| via central difference: scale ALL body
    matrices by e^±δ, measure (L(+)−L(−))/2δ. Should ≈ Σ ⟨g,W⟩ if the anti-radial
    gradient is a real loss derivative. Restores weights."""
    bodies = [p for n, p in raw.named_parameters() if _isbody(n)]
    def scale_all(c):
        for p in bodies:
            _local(p).mul_(c)
    scale_all(math.exp(delta))
    Lp = _ce_loss(model, x, y, pad_id, raw).item()
    scale_all(math.exp(-2*delta))      # now at e^-δ relative to original
    Lm = _ce_loss(model, x, y, pad_id, raw).item()
    scale_all(math.exp(delta))         # restore
    return (Lp - Lm) / (2*delta)       # ≈ dL/dlog(scale) summed over body


def run(ckpt, groups_override=None, config_path=None, ntokens=2048, seq=1024,
        shard_strategy="none", out_path=None, seed=0):
    device = nc.detect_device(None)
    path = resolve_ckpt(ckpt)
    step = int(re.search(r"_(\d+)\.pt", os.path.basename(path)).group(1)) \
        if re.search(r"_(\d+)\.pt", os.path.basename(path)) else None
    logger.print_and_log(f"=== KEEL radial probe: {os.path.basename(path)} on {device} ===")
    t0 = time.time()
    model, enc, cfg = nc.load_model_and_tokenizer(
        path, device=device, half_precision=True, shard_strategy=shard_strategy, use_keel=None)
    pad_id = int(getattr(cfg, "pad_id", 0) or 0)
    logger.print_and_log(f"loaded in {time.time()-t0:.1f}s; pad_id={pad_id}")
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model

    data_root = getattr(cfg, "data_root_path", "../../notebooks/datasets/tokenized/llama/")
    if groups_override:
        groups, gsrc = groups_override, "CLI override"
    else:
        groups, gsrc = _resolve_own_groups(cfg, path, config_path=config_path)
    if not groups:
        raise RuntimeError("could not determine groups; pass --config or --groups")
    logger.print_and_log(f"groups [{gsrc}]: {groups}")
    win = min(seq, int(getattr(getattr(model, "params", None), "max_seq_len", seq) or seq))
    tokens, targets = build_panel(data_root, groups, win + 1, device, seed=seed)
    x = tokens[:, :win]; y = targets[:win]
    import statistics as st
    result = {'checkpoint': os.path.basename(path), 'step': step}

    def med(dd):
        v = list(dd.values())
        return (st.median(v), st.mean(v), sum(1 for c in v if c < 0)/len(v)) if v else (None,)*3

    # ---- PROBE 1: train vs eval radial gradient + finite diff ----
    logger.print_and_log("\n=== PROBE 1: radial gradient, eval vs train forward ===")
    model.eval()
    cos_eval, L_eval = _body_grad_cos(model, x, y, pad_id, raw)
    model.train()
    cos_train, L_train = _body_grad_cos(model, x, y, pad_id, raw)
    me = med(cos_eval); mt = med(cos_train)
    logger.print_and_log(f"  EVAL  forward: cos(g,W) median={me[0]:+.5f} mean={me[1]:+.5f} negfrac={me[2]*100:.0f}%  (L={L_eval:.4f})")
    logger.print_and_log(f"  TRAIN forward: cos(g,W) median={mt[0]:+.5f} mean={mt[1]:+.5f} negfrac={mt[2]*100:.0f}%  (L={L_train:.4f})")
    # finite-diff under both modes
    model.eval(); fd_eval = _finite_diff_radial(model, x, y, pad_id, raw)
    model.train(); fd_train = _finite_diff_radial(model, x, y, pad_id, raw)
    # compare to sum of <g,W> from the train grad
    for p in raw.parameters(): p.grad = None
    model.train(); _ce_loss(model, x, y, pad_id, raw).backward()
    sum_gW = sum(_gsum((_local(p).float()*_local(p.grad).float()).sum(), p).item()
                 for n, p in raw.named_parameters() if _isbody(n) and p.grad is not None)
    logger.print_and_log(f"  finite-diff dL/dlog(body scale): eval={fd_eval:+.5f} train={fd_train:+.5f}")
    logger.print_and_log(f"  sum<g,W> (train grad) = {sum_gW:+.5f}  -> finite-diff(train) should match this")
    result['probe1'] = {'cos_eval': me, 'cos_train': mt, 'L_eval': L_eval, 'L_train': L_train,
                        'finite_diff_eval': fd_eval, 'finite_diff_train': fd_train, 'sum_gW_train': sum_gW}

    # ---- PROBE 2: branch-gain derivative dL/dlog g_l ----
    logger.print_and_log("\n=== PROBE 2: branch-gain derivative dL/d log g_l (g_l=1) ===")
    model.train()
    gains = {}   # (layer_idx, 'attn'|'ffn') -> scalar leaf
    handles = []
    def mk_hook(key):
        def hook(_m, _inp, out):
            g = gains[key]
            return out * g if not isinstance(out, (tuple, list)) else (out[0]*g, *out[1:])
        return hook
    for i, blk in enumerate(raw.layers):
        for sub, mod in (('attn', getattr(blk, 'attention', None)), ('ffn', getattr(blk, 'feed_forward', None))):
            if mod is None:
                continue
            key = (i, sub)
            gains[key] = torch.ones((), device=device, requires_grad=True)
            handles.append(mod.register_forward_hook(mk_hook(key)))
    for p in raw.parameters(): p.grad = None
    loss = _ce_loss(model, x, y, pad_id, raw)
    gl = list(gains.values())
    grads = torch.autograd.grad(loss, gl, retain_graph=False, allow_unused=True)
    for h in handles: h.remove()
    # dL/dlog g_l = g_l * dL/dg_l ; at g_l=1 it's just dL/dg_l
    dlogg = {k: (gr.item() if gr is not None else 0.0) for k, gr in zip(gains.keys(), grads)}
    attn_d = [v for (i, s), v in dlogg.items() if s == 'attn']
    ffn_d = [v for (i, s), v in dlogg.items() if s == 'ffn']
    if attn_d:
        logger.print_and_log(f"  attn branch dL/dlog g: median={st.median(attn_d):+.5f} mean={st.mean(attn_d):+.5f} negfrac={sum(1 for v in attn_d if v<0)/len(attn_d)*100:.0f}%")
    if ffn_d:
        logger.print_and_log(f"  ffn  branch dL/dlog g: median={st.median(ffn_d):+.5f} mean={st.mean(ffn_d):+.5f} negfrac={sum(1 for v in ffn_d if v<0)/len(ffn_d)*100:.0f}%")
    logger.print_and_log("  NEGATIVE => model wants MORE branch relative to highway => explains anti-radial body grad.")
    result['probe2'] = {'attn_dlogg': {str(i): v for (i, s), v in dlogg.items() if s == 'attn'},
                        'ffn_dlogg': {str(i): v for (i, s), v in dlogg.items() if s == 'ffn'},
                        'attn_median': st.median(attn_d) if attn_d else None,
                        'ffn_median': st.median(ffn_d) if ffn_d else None}

    # ---- PROBE 3: epsilon sensitivity ----
    logger.print_and_log("\n=== PROBE 3: RMSNorm-eps sensitivity of cos(g,W) (train forward) ===")
    from model_v2 import RMSNorm
    norms = [m for m in raw.modules() if isinstance(m, RMSNorm)]
    orig_eps = [getattr(m, 'eps', None) for m in norms]
    eps_rows = {}
    model.train()
    for eps in (1e-5, 1e-6, 1e-8, 0.0):
        for m in norms:
            if hasattr(m, 'eps'):
                m.eps = eps
        cos_e, _ = _body_grad_cos(model, x, y, pad_id, raw)
        m_ = med(cos_e)
        eps_rows[str(eps)] = {'median': m_[0], 'mean': m_[1], 'negfrac': m_[2]}
        logger.print_and_log(f"  eps={eps:.0e}: cos(g,W) median={m_[0]:+.5f} negfrac={m_[2]*100:.0f}%")
    for m, e in zip(norms, orig_eps):
        if e is not None:
            m.eps = e
    logger.print_and_log("  cos -> 0 as eps -> 0  => eps is the source; persists => branch/highway mixing.")
    result['probe3'] = eps_rows

    if out_path:
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=1)
        logger.print_and_log(f"\nwrote {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--ntokens", type=int, default=2048)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--shard", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    go = [g.strip() for g in a.groups.split(",")] if a.groups else None
    run(a.ckpt, groups_override=go, config_path=a.config, ntokens=a.ntokens, seq=a.seq,
        shard_strategy=a.shard, out_path=a.out, seed=a.seed)


if __name__ == "__main__":
    main()
