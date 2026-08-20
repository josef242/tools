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


# ============================================================================
# STAGE B — corrected train-branch vs eval-branch radial-gradient comparison
# ============================================================================
# The original PROBE 1 had a bug: `_ce_loss` ALWAYS used the eval branch
# (`model(x)` -> output(h) -> external F.cross_entropy.float()), so "train vs
# eval" was identical by construction and it NEVER exercised the train-branch
# fused-CCE. That reproduced Part B's +0.0003 sign-random null but not the
# in-situ -0.0129. This stage fixes it: it backwards the ACTUAL train-branch
# fused-CCE (cce_loss == linear_cross_entropy on h_flat @ output.weight) and
# compares cos(g,W) head-to-head against the eval-branch external CE on the
# SAME batch, under the SAME autocast(bf16) the trainer uses.
#
# The discriminator for Josef's question ("is the growth purely a bf16-CCE
# artifact?"): the real run accumulates CCE in bf16 (output.weight is bf16 =>
# accum_*_fp32 = (out_dtype==fp32) = False). The `accum_fp32` cell forces the
# fused kernel's internal e/c accumulation to fp32 (the flag the code already
# supports). If train-CCE-bf16 = -0.0129 but train-CCE-fp32accum ~ 0, the lean
# is a bf16-accumulation artifact, not loss geometry.
#
# Three loss paths, identical batch, identical model.train() mode:
#   eval_ce      : model(x) -> output(h).float() -> F.cross_entropy   (Part B path)
#   train_cce    : model(x, y) -> (None, fused cce_loss bf16-accum)    (in-situ path)
#   train_cce_f32: cce_loss with accum_e_fp32=accum_c_fp32=True        (discriminator)
# Run at multiple context lengths to separate kernel (1) from long-context (2).

def _train_branch_loss(raw, x, y, pad_id, accum_fp32_override=None):
    """Backward the model's ACTUAL train-branch fused-CCE. Calls forward with
    targets so the model returns (None, loss) where loss == cce_loss(...). If
    accum_fp32_override is True/False it temporarily forces the kernel accum
    dtype regardless of weight dtype (the bf16-vs-fp32 discriminator)."""
    # The model decides accum from out_dtype==fp32 internally; to force it we
    # monkeypatch cce_loss's kwargs via the model module. Cleanest: call the
    # same path the model uses, replicating its h then cce_loss, but to stay
    # faithful to the production forward we instead let the model compute loss
    # natively for the default case, and only override for the f32 cell by
    # re-deriving from h (captured via a forward hook on self.norm).
    if accum_fp32_override is None:
        out = raw(x, y, active_layers=None, scaffold_mode=False)
        loss = out[1] if isinstance(out, (tuple, list)) else out
        return loss
    # Forced-accum cell: capture h after final norm, recompute cce_loss with
    # the chosen accum flags — same kernel, same h, only the accum dtype differs.
    import model_v2 as _mv
    cap = {}
    def _hook(_m, _i, o):
        cap['h'] = o
    hnd = raw.norm.register_forward_hook(_hook)
    try:
        _ = raw(x, y, active_layers=None, scaffold_mode=False)
    finally:
        hnd.remove()
    h = cap['h']
    h_flat = h.reshape(-1, h.size(-1))
    od = raw.output.weight.dtype
    if h_flat.dtype != od:
        h_flat = h_flat.to(od)
    loss = _mv.cce_loss(
        h_flat, raw.output.weight, y.reshape(-1),
        accum_e_fp32=accum_fp32_override, accum_c_fp32=accum_fp32_override,
        reduction="mean", ignore_index=pad_id,
    )
    return loss


def _grad_cos_from_loss(raw, loss_fn):
    """Zero grads, run loss_fn() -> scalar loss, backward, return
    ({name: cos(g,W)} for body matrices, loss_value)."""
    for p in raw.parameters():
        p.grad = None
    loss = loss_fn()
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
    return out, float(loss.item())


def run_stageb(ckpt, groups_override=None, config_path=None, seqs=(1024,),
               shard_strategy="none", out_path=None, seed=0, dtype="bf16",
               tokens_file=None):
    """Corrected train-CCE vs eval-CE radial-gradient comparison + bf16/fp32
    accum discriminator, across context lengths.

    tokens_file: if given, REPLAY the exact captured tokens (raw binary {'x','y'}
    from WD_DUMP_TOKENS) instead of build_panel — splits the in-situ residual into
    real-stream-DATA vs FSDP/bf16-PATH (Math Agent #1). BLACK BOX: tokens are loaded
    as ints and fed straight to the model; never decoded/printed/logged (shapes only).
    Overrides seqs (uses the captured T)."""
    import statistics as st
    device = nc.detect_device(None)
    path = resolve_ckpt(ckpt)
    step = int(re.search(r"_(\d+)\.pt", os.path.basename(path)).group(1)) \
        if re.search(r"_(\d+)\.pt", os.path.basename(path)) else None
    logger.print_and_log(f"=== KEEL Stage B (train-CCE vs eval-CE): {os.path.basename(path)} on {device} ===")
    t0 = time.time()
    model, enc, cfg = nc.load_model_and_tokenizer(
        path, device=device, half_precision=True, shard_strategy=shard_strategy, use_keel=None)
    pad_id = int(getattr(cfg, "pad_id", 0) or 0)
    logger.print_and_log(f"loaded in {time.time()-t0:.1f}s; pad_id={pad_id}")
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw.train()  # match production: train-branch forward, dropout=0 so backbone is identical
    # The offline loader sets block use_activation_checkpointing=False (eval doesn't
    # need it). Production trains WITH act-ckpt (config use_activation_checkpointing=
    # true), and at long context the full backward graph OOMs a single 4080 without
    # it. Force it on per-block: makes long-T fit AND matches the production backward
    # (act-ckpt recompute was itself a flagged confound — enabling it tests faithfully).
    _nac = 0
    for _blk in getattr(raw, 'layers', []):
        if hasattr(_blk, 'use_activation_checkpointing'):
            _blk.use_activation_checkpointing = True
            _nac += 1
    logger.print_and_log(f"forced activation checkpointing ON for {_nac} blocks (match production backward)")

    data_root = getattr(cfg, "data_root_path", "../../notebooks/datasets/tokenized/llama/")
    if groups_override:
        groups, gsrc = groups_override, "CLI override"
    else:
        groups, gsrc = _resolve_own_groups(cfg, path, config_path=config_path)
    if not groups:
        raise RuntimeError("could not determine groups; pass --config or --groups")
    logger.print_and_log(f"groups [{gsrc}]: {groups}")

    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    max_seq = int(getattr(getattr(model, "params", None), "max_seq_len", max(seqs)) or max(seqs))

    def med(dd):
        v = list(dd.values())
        return (st.median(v), st.mean(v), sum(1 for c in v if c < 0)/len(v), len(v)) if v else (None,)*4

    # REPLAY mode: load exact captured tokens (opaque ints), use their T.
    _replay = None
    if tokens_file:
        _d = torch.load(tokens_file, map_location=device)
        _rx = _d['x'].to(device); _ry = _d['y'].to(device)
        _replay = (_rx, _ry)
        seqs = (int(_rx.shape[-1]),)  # use captured T
        logger.print_and_log(f"REPLAY exact captured tokens (shape {tuple(_rx.shape)}, OPAQUE) from {os.path.basename(tokens_file)}")

    result = {'checkpoint': os.path.basename(path), 'step': step, 'dtype': dtype,
              'replay': bool(tokens_file), 'by_seq': {}}
    for seq in seqs:
        win = min(seq, max_seq)
        if win < seq:
            logger.print_and_log(f"  [warn] requested seq {seq} > max_seq_len {max_seq}; clamping to {win}")
        if _replay is not None:
            # exact captured tokens; y may be [N] flat or [1,N] — normalise to model's
            # train-branch expectation (x=[1,win], y=[win] flat). NEVER inspect content.
            _rx, _ry = _replay
            x = _rx[:, :win] if _rx.dim() == 2 else _rx[:win].unsqueeze(0)
            y = (_ry.reshape(-1))[:win]
        else:
            tokens, targets = build_panel(data_root, groups, win + 1, device, seed=seed)
            x = tokens[:, :win]; y = targets[:win]
        logger.print_and_log(f"\n--- T={win} (batch [1,{win}]){' REPLAY' if _replay is not None else ''} ---")

        # the three loss paths, same batch, same train() mode, same autocast
        def eval_ce():
            out = raw(x)  # eval branch: targets=None -> (logits, None)
            logits = (out[0] if isinstance(out, (tuple, list)) else out)
            logits = logits.reshape(-1, raw.output.weight.shape[0]).float()
            return F.cross_entropy(logits, y.reshape(-1), ignore_index=pad_id)
        def train_cce():
            return _train_branch_loss(raw, x, y, pad_id, accum_fp32_override=None)
        def train_cce_f32():
            return _train_branch_loss(raw, x, y, pad_id, accum_fp32_override=True)

        seq_rows = {}
        for tag, fn in (('eval_ce', eval_ce), ('train_cce_bf16', train_cce), ('train_cce_f32accum', train_cce_f32)):
            try:
                with torch.autocast(device_type=device if isinstance(device, str) else device.type,
                                    dtype=autocast_dtype):
                    cos, L = _grad_cos_from_loss(raw, fn)
                m = med(cos)
                seq_rows[tag] = {'median': m[0], 'mean': m[1], 'negfrac': m[2], 'n': m[3], 'loss': L,
                                 'cos': {n: c for n, c in cos.items()}}
                logger.print_and_log(
                    f"  {tag:18s}: cos(g,W) median={m[0]:+.5f} mean={m[1]:+.5f} "
                    f"negfrac={m[2]*100:3.0f}% n={m[3]} (L={L:.4f})")
            except Exception as e:
                seq_rows[tag] = {'error': f"{type(e).__name__}: {e}"}
                logger.print_and_log(f"  {tag:18s}: ERROR {type(e).__name__}: {e}")
        result['by_seq'][str(win)] = seq_rows

    # verdict helper
    logger.print_and_log("\n=== STAGE B VERDICT ===")
    for sk, rows in result['by_seq'].items():
        e = rows.get('eval_ce', {}).get('median')
        tb = rows.get('train_cce_bf16', {}).get('median')
        tf = rows.get('train_cce_f32accum', {}).get('median')
        logger.print_and_log(f"  T={sk}: eval_ce={e} train_cce_bf16={tb} train_cce_f32accum={tf}")
    logger.print_and_log("  Read: if train_cce_bf16 << 0 but eval_ce ~ 0 -> KERNEL is the source.")
    logger.print_and_log("        if train_cce_f32accum ~ eval_ce (~0) -> bf16-ACCUM artifact (Josef's hypothesis).")
    logger.print_and_log("        if train_cce_f32accum still << 0 -> real fused-CE geometry, not rounding.")

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
    ap.add_argument("--mode", default="legacy", choices=["legacy", "stageb"],
                    help="legacy = original 3 probes; stageb = corrected train-CCE vs eval-CE + accum discriminator")
    ap.add_argument("--seqs", default=None,
                    help="stageb only: comma-sep context lengths, e.g. 1024,12288 (default: --seq)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="stageb only: autocast dtype (match the run's data_type; mf=bf16)")
    ap.add_argument("--tokens-file", default=None,
                    help="stageb only: REPLAY exact captured tokens (raw binary from WD_DUMP_TOKENS) "
                         "instead of build_panel — splits residual into data vs path. Tokens stay OPAQUE.")
    a = ap.parse_args()
    go = [g.strip() for g in a.groups.split(",")] if a.groups else None
    if a.mode == "stageb":
        seqs = tuple(int(s) for s in a.seqs.split(",")) if a.seqs else (a.seq,)
        run_stageb(a.ckpt, groups_override=go, config_path=a.config, seqs=seqs,
                   shard_strategy=a.shard, out_path=a.out, seed=a.seed, dtype=a.dtype,
                   tokens_file=a.tokens_file)
    else:
        run(a.ckpt, groups_override=go, config_path=a.config, ntokens=a.ntokens, seq=a.seq,
            shard_strategy=a.shard, out_path=a.out, seed=a.seed)


if __name__ == "__main__":
    main()
