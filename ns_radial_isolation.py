"""Isolated Newton-Schulz radial-injection test (NO FSDP, NO trainer, NO data).

Settles: does the −0.0129 radial component come from the MATH of Newton-Schulz
orthogonalization (intrinsic, any correct impl) or from a downstream stage of THIS
NorMuon implementation (apply_scaling / apply_normuon)?

Construct a RADIAL-NULL gradient G ⟂ W (project out the radial part so ⟨G,W⟩=0 exactly),
then push it through the real muon_fsdp2 stages one at a time and measure cos(stage, W):
  raw G (≈0 by construction) -> NS -> scaling -> normuon
If cos jumps to ~−0.013 right after NS  => intrinsic to Newton-Schulz (the matrix-sign /
polar factor genuinely leans anti-radial vs W). If it stays ~0 through NS but jumps at
apply_normuon => it's the NorMuon rescale (implementation-specific). Etc.

Uses REAL body W matrices from a checkpoint so shapes/scale are realistic. Random G seed
varied per matrix. Pure single-card fp32 math.

Run: python ns_radial_isolation.py --ckpt <pt> [--n 40] [--seed 0]
"""
import os, sys, re, math, argparse, json, statistics as st
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
from muon_fsdp2 import (zeropower_via_newtonschulz5 as ns,
                        apply_scaling, apply_normuon)  # noqa: E402


def _cos(a, b):
    a = a.float(); b = b.float()
    d = (a * b).sum().item()
    an = a.norm().item(); bn = b.norm().item()
    return (d / (an * bn)) if (an > 0 and bn > 0) else 0.0


def _isbody(n):
    return any(n.endswith(s) for s in ('wo.weight', 'w2.weight', 'wq.weight',
                                       'wk.weight', 'wv.weight', 'w1.weight', 'w3.weight'))


def run(ckpt, n=40, seed=0, ns_steps=5, beta2=0.99):
    dev = nc.detect_device(None)
    path = resolve_ckpt(ckpt)
    logger.print_and_log(f"=== NS radial-isolation: {os.path.basename(path)} on {dev} ===")
    model, enc, cfg = nc.load_model_and_tokenizer(
        path, device=dev, half_precision=True, shard_strategy="none", use_keel=None)
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model

    bodies = [(nm, p) for nm, p in raw.named_parameters() if _isbody(nm) and p.dim() == 2]
    logger.print_and_log(f"{len(bodies)} body matrices; sampling {min(n, len(bodies))}")
    g = torch.Generator(device=dev).manual_seed(seed)

    rows = {'rawG': [], 'afterNS': [], 'afterScale': [], 'afterNormuon': []}
    for i, (nm, p) in enumerate(bodies[:n]):
        W = p.detach().float()
        # radial-null random gradient: G ⟂ W  (project out the radial component)
        G = torch.randn(W.shape, generator=g, device=dev, dtype=torch.float32)
        G = G - W * ((G * W).sum() / (W * W).sum())   # now ⟨G,W⟩ = 0 exactly
        rows['rawG'].append(_cos(G, W))               # ≈ 0 by construction

        u = ns(G.clone(), ns_steps).float()
        rows['afterNS'].append(_cos(u, W))

        u2 = apply_scaling(u.clone(), rms_scale=False)
        rows['afterScale'].append(_cos(u2, W))

        # NorMuon needs a second-moment buffer; start cold (zeros) — tests the rescale shape
        sm = torch.zeros((W.shape[0], 1), device=dev, dtype=torch.float32)
        u3 = apply_normuon(u2.clone(), sm, beta2)
        rows['afterNormuon'].append(_cos(u3, W))

    def summ(v):
        return dict(median=st.median(v), mean=st.mean(v),
                    negfrac=sum(1 for c in v if c < 0) / len(v), n=len(v))

    logger.print_and_log("\n=== cos(stage, W) for a RADIAL-NULL input gradient G ⟂ W ===")
    out = {}
    for k in ('rawG', 'afterNS', 'afterScale', 'afterNormuon'):
        s = summ(rows[k]); out[k] = s
        logger.print_and_log(f"  {k:14s}: median={s['median']:+.5f} mean={s['mean']:+.5f} "
                             f"negfrac={s['negfrac']*100:3.0f}% n={s['n']}")
    logger.print_and_log(
        "\nRead: rawG≈0 (by construction). If afterNS<<0 => INTRINSIC to Newton-Schulz "
        "(polar factor leans anti-radial vs W). If afterNS≈0 but a later stage<<0 => that "
        "stage (scaling/normuon) is the source (implementation-specific).")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ns-steps", type=int, default=5)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.ckpt, n=a.n, seed=a.seed, ns_steps=a.ns_steps, beta2=a.beta2)
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)
        logger.print_and_log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
