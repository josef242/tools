"""Probe A — clip-replay (Math Agent Brief #5). NO trainer, NO FSDP, NO optimizer.step, NO data.

Question: KeelHaul's loss lags DN2 and `nrm` (raw pre-clip global grad norm) is high vs clip=1.0.
The Math Agent's leading hypothesis: the GLOBAL clip is set by the large Muon-BODY raw gradient,
but the rescale ALSO hits the Adam groups (head/emb/norm/router) which DO use magnitude -> those
get throttled, while the Muon body barely cares (Newton-Schulz discards magnitude). And with WARM
momentum, a variable clip coeff can alter the body update DIRECTION (momentum staleness), even
though magnitude is untouched.

This probe replays the clip at scales c in {1.0, 0.75, 0.5, 0.25, 0.1} and pushes c*G through the
REAL muon_fsdp2 transform (momentum -> Newton-Schulz -> apply_scaling -> NorMuon -> tangent proj),
and the REAL adam_update, measuring how much each group's UPDATE actually changes.

Read on results:
  BODY (Muon): if cos(U_c, U_1)~1 and ||U_c||/||U_1||~1 across c  => clip is a ~no-op on body (cold).
               Then the COLD-vs-WARM gap shows the momentum-staleness effect.
  ADAM:        if ||D_c||/||D_1|| shrinks materially with c     => clip THROTTLES Adam groups.
  => If body invariant but Adam throttled, the diagnosis is closed: split the clip (per-group).

Uses REAL W matrices from a checkpoint for realistic shapes/scale. Gradients are realistic-magnitude
random (the property under test is the transform's RESPONSE TO SCALING c, which is gradient-content
independent — polar(cG)=polar(G) — so synthetic G is valid and avoids any data handling).
PURE single-card fp32 math. Read-only: never steps the optimizer, never writes weights.

Run: python clip_replay_probe.py --ckpt <pt> [--n 48] [--seed 0] [--out clip_replay.json]
"""
import os, sys, math, argparse, json, statistics as st
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("../common_fsdp2", "../saved_code"):
    _ap = os.path.normpath(os.path.join(_HERE, _p))
    if _ap not in sys.path:
        sys.path.insert(0, _ap)
import logger  # noqa: E402
logger._instance.set_logdir("./logs"); logger._instance.set_rank(0)
import neo_common as nc  # noqa: E402
from zloss_row_center_probe import resolve_ckpt  # noqa: E402
from muon_fsdp2 import (zeropower_via_newtonschulz5 as ns, apply_momentum,
                        apply_scaling, apply_normuon, adam_update)  # noqa: E402

CLIP_SCALES = [1.0, 0.75, 0.5, 0.25, 0.1]


def _cos(a, b):
    a = a.float().reshape(-1); b = b.float().reshape(-1)
    d = (a * b).sum().item(); an = a.norm().item(); bn = b.norm().item()
    return (d / (an * bn)) if (an > 0 and bn > 0) else 0.0


def _isbody(n):
    return any(n.endswith(s) for s in ('wo.weight', 'w2.weight', 'wq.weight',
                                       'wk.weight', 'wv.weight', 'w1.weight', 'w3.weight'))


def _isadam(n):
    # the magnitude-sensitive groups: embeddings, output head, norms, router, GDN small params
    return any(s in n for s in ('tok_embeddings', 'output.weight', '_norm', 'norm.weight',
                                'router', 'a_proj', 'b_proj', 'conv1d'))


def muon_transform(G, W, momentum_buf, beta, ns_steps, beta2, rms_scale,
                   tangent_project=True):
    """The REAL Muon body transform on a single 2D matrix, mirroring Fsdp1dWork.finish:
    momentum -> NS -> apply_scaling -> NorMuon -> tangent projection. Returns the update U
    (pre-lr). momentum_buf is mutated (lerp), so pass a fresh clone per call."""
    update = apply_momentum(G.clone(), momentum_buf, beta, nesterov=False)
    update = ns(update, ns_steps).float()
    update = apply_scaling(update, rms_scale=rms_scale)
    sm = torch.zeros((update.shape[0], 1), device=update.device, dtype=torch.float32)
    update = apply_normuon(update, sm, beta2)
    if tangent_project:
        uf = update.reshape(-1).float(); wf = W.reshape(-1).float()
        dot = (uf * wf).sum(); wsq = (wf * wf).sum()
        if wsq.item() > 0:
            update = update - W * (dot / wsq)   # preserve_norm=false (KeelHaul setting)
    return update


def run(ckpt, n=48, seed=0, ns_steps=5, beta=0.95, beta2=0.99, adam_beta=(0.9, 0.95),
        rms_scale=True, gpu=None):
    dev = nc.detect_device(gpu) if gpu is not None else nc.detect_device(None)
    path = resolve_ckpt(ckpt)
    logger.print_and_log(f"=== Probe A clip-replay: {os.path.basename(path)} on {dev} ===")
    logger.print_and_log(f"    momentum_beta={beta} ns_steps={ns_steps} normuon_beta2={beta2} "
                         f"rms_scale={rms_scale} clip_scales={CLIP_SCALES}")
    model, enc, cfg = nc.load_model_and_tokenizer(
        path, device=dev, half_precision=True, shard_strategy="none", use_keel=None)
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model

    bodies = [(nm, p) for nm, p in raw.named_parameters() if _isbody(nm) and p.dim() == 2]
    adams = [(nm, p) for nm, p in raw.named_parameters() if _isadam(nm) and p.dim() >= 1]
    logger.print_and_log(f"{len(bodies)} body (Muon) matrices, {len(adams)} Adam-group params; "
                         f"sampling {min(n, len(bodies))} / {min(n, len(adams))}")
    g = torch.Generator(device=dev).manual_seed(seed)

    # ---- BODY (Muon): cold vs warm momentum, per clip scale ----
    # store cos(U_c,U_1), ||U_c||/||U_1||, cos(U_c,W) for each scale, separately cold & warm
    body = {m: {c: {'cosU': [], 'magU': [], 'cosW': []} for c in CLIP_SCALES}
            for m in ('cold', 'warm')}
    for nm, p in bodies[:n]:
        W = p.detach().float()
        # realistic-magnitude gradient: scale so ||G||_rms ~ O(1) like a real body grad
        G = torch.randn(W.shape, generator=g, device=dev, dtype=torch.float32)
        G = G / (G.norm() + 1e-9) * (W.numel() ** 0.5) * 0.05   # ~5% of a unit-RMS field
        # a realistic WARM momentum buffer: correlated-but-not-identical to G (stale history)
        Mwarm0 = 0.8 * G + 0.6 * torch.randn(W.shape, generator=g, device=dev, dtype=torch.float32) \
                              / (W.numel() ** 0.5) ** 0 * G.norm() / (W.numel() ** 0.5)
        for mode, M0 in (('cold', torch.zeros_like(G)), ('warm', Mwarm0)):
            U1 = muon_transform((1.0 * G), W, M0.clone(), beta, ns_steps, beta2, rms_scale)
            for c in CLIP_SCALES:
                Uc = muon_transform((c * G), W, M0.clone(), beta, ns_steps, beta2, rms_scale)
                body[mode][c]['cosU'].append(_cos(Uc, U1))
                body[mode][c]['magU'].append((Uc.norm() / (U1.norm() + 1e-12)).item())
                body[mode][c]['cosW'].append(_cos(Uc, W))

    # ---- ADAM groups: ||D_c||/||D_1|| and cos(D_c,D_1) ----
    # Adam at warm-ish state (nonzero exp_avg / exp_avg_sq), per clip scale.
    adam = {c: {'magD': [], 'cosD': []} for c in CLIP_SCALES}
    for nm, p in adams[:n]:
        P = p.detach().float()
        G = torch.randn(P.shape, generator=g, device=dev, dtype=torch.float32)
        G = G / (G.norm() + 1e-9) * (P.numel() ** 0.5) * 0.05
        # warm Adam state: exp_avg ~ correlated with G, exp_avg_sq ~ G^2 scale
        ea0 = 0.9 * G + 0.1 * torch.randn(P.shape, generator=g, device=dev, dtype=torch.float32) \
                            / (P.numel() ** 0.5) * G.norm()
        es0 = (G * G).mean().expand_as(G).clone() * 0.5 + (G * G) * 0.5
        def adam_d(gc):
            ea = ea0.clone(); es = es0.clone()
            return adam_update(gc, ea, es, step=200, betas=adam_beta, eps=1e-8)
        D1 = adam_d(1.0 * G)
        for c in CLIP_SCALES:
            Dc = adam_d(c * G)
            adam[c]['magD'].append((Dc.norm() / (D1.norm() + 1e-12)).item())
            adam[c]['cosD'].append(_cos(Dc, D1))

    # ---- report ----
    def med(v):
        return st.median(v) if v else float('nan')

    out = {'ckpt': os.path.basename(path), 'clip_scales': CLIP_SCALES,
           'body': {}, 'adam': {}}
    logger.print_and_log("\n=== BODY (Muon) — does clipping change the body UPDATE? ===")
    logger.print_and_log("(cos(U_c,U_1)=1 & mag=1  => clip is a no-op on the body)")
    for mode in ('cold', 'warm'):
        logger.print_and_log(f"  -- {mode} momentum --")
        logger.print_and_log(f"     {'c':>5} | {'cos(U_c,U_1)':>13} | {'||U_c||/||U_1||':>15} | {'cos(U_c,W)':>11}")
        out['body'][mode] = {}
        for c in CLIP_SCALES:
            cu = med(body[mode][c]['cosU']); mu = med(body[mode][c]['magU']); cw = med(body[mode][c]['cosW'])
            out['body'][mode][c] = {'cosU': cu, 'magU': mu, 'cosW': cw}
            logger.print_and_log(f"     {c:>5} | {cu:>+13.5f} | {mu:>15.5f} | {cw:>+11.5f}")

    logger.print_and_log("\n=== ADAM groups — does clipping throttle the magnitude-sensitive params? ===")
    logger.print_and_log("(||D_c||/||D_1|| << 1 as c shrinks  => clip THROTTLES Adam updates)")
    logger.print_and_log(f"     {'c':>5} | {'||D_c||/||D_1||':>15} | {'cos(D_c,D_1)':>13}")
    for c in CLIP_SCALES:
        mg = med(adam[c]['magD']); cd = med(adam[c]['cosD'])
        out['adam'][c] = {'magD': mg, 'cosD': cd}
        logger.print_and_log(f"     {c:>5} | {mg:>15.5f} | {cd:>+13.5f}")

    # verdict heuristic
    body_inv = all(abs(out['body']['cold'][c]['cosU'] - 1) < 0.02 and
                   abs(out['body']['cold'][c]['magU'] - 1) < 0.05 for c in CLIP_SCALES)
    warm_dev = max(1 - out['body']['warm'][c]['cosU'] for c in CLIP_SCALES)
    adam_throttle = out['adam'][0.1]['magD']  # how much Adam update survives at c=0.1
    logger.print_and_log("\n=== VERDICT ===")
    logger.print_and_log(f"  body cold-invariant to clip?  {body_inv}  "
                         f"(cos~1 & mag~1 across c => clip no-op on body magnitude+direction)")
    logger.print_and_log(f"  warm-momentum direction shift: up to {warm_dev:+.4f} drop in cos(U_c,U_1) "
                         f"(momentum-staleness effect; >~0.02 means clipping alters body direction)")
    logger.print_and_log(f"  Adam update surviving at c=0.1: {adam_throttle:.3f}x "
                         f"(<<1 => clip throttles Adam groups — the leading hypothesis)")
    out['verdict'] = {'body_cold_invariant': body_inv, 'warm_max_dir_shift': warm_dev,
                      'adam_mag_at_c0.1': adam_throttle}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ns-steps", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.95, help="muon momentum beta")
    ap.add_argument("--beta2", type=float, default=0.99, help="normuon beta2")
    ap.add_argument("--gpu", type=int, default=None, help="CUDA ordinal (cuda:0 is often the biggest card)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.ckpt, n=a.n, seed=a.seed, ns_steps=a.ns_steps, beta=a.beta, beta2=a.beta2, gpu=a.gpu)
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)
        logger.print_and_log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
