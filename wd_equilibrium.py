"""WD-equilibrium analysis (Test 2 of the taming battery; Math Agent equilibrium model).

For a pre-norm (scale-invariant) matrix under decoupled WD + a tangential optimizer
update U_t ⟂ W_t:
    ||W_{t+1}||^2 ≈ (1-ηλ)^2 ||W_t||^2 + ||U_t||^2
Equilibrium (||W|| stops growing) when radial WD removal balances tangential
injection:
    2ηλ ||W_eq||^2 ≈ ||U||^2   =>   ||W_eq|| ≈ ||U|| / sqrt(2ηλ)
where ||U|| = eff_lr * ||update|| is the per-STEP weight change magnitude.

This is the TAMING DIAL: given the measured per-step update magnitude (||update||
from the NorMuon-decomp probe) and learning rate eta, compute the equilibrium ||W||
that each WD strength lambda would produce — i.e. "what WD bounds the body norm at
a target scale?" Compares to current ||W|| (from diagnostics.jsonl) to show how far
above/below equilibrium each layer currently is, and what lambda would pin it near
its CURRENT (or a target) norm.

INPUTS (all already on disk, no GPU):
  --nud   nud_<ckpt>.json   (per-matrix update_norm from normuon_update_decomp.py)
  --diag  diagnostics.jsonl (per-layer current w_norm)
  --eta   current LR (eta); --wd current lambda (0.02)

NOTE on ||U|| = eff_lr * ||update||: the nud probe reports ||update|| (the NorMuon
update DIRECTION magnitude, post apply_normuon which preserves pre-norm magnitude).
Multiply by eff_lr to get the actual per-step ||ΔW||. apply_normuon keeps update
norm ~= the orthogonalized-grad norm, so ||U|| is roughly LR-scaled and only weakly
gradient-dependent — which is WHY the ramp is LR-coupled (Part A) and why equilibrium
scales as ||U||/sqrt(2ηλ) ∝ eta^{1/2}/sqrt(lambda).
"""
import os
import sys
import json
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wd_waste_probe import _load_jsonl  # tolerant jsonl reader


def _class_of(name):
    if name.endswith('attention.wo.weight') or name.endswith('feed_forward.w2.weight'):
        return 'body_proj'
    if any(name.endswith(s) for s in ('attention.wq.weight','attention.wk.weight',
                                      'attention.wv.weight','feed_forward.w1.weight','feed_forward.w3.weight')):
        return 'body_in'
    if name == 'output.weight':
        return 'head'
    if 'tok_embeddings' in name:
        return 'embedding'
    return 'other'


def run(nud_path, diag_path, eta, wd, lambda_mults=(1, 2, 4, 10), out_path=None):
    nud = json.load(open(nud_path))
    rows = nud['per_matrix']
    # current w_norm: prefer the nud probe's own w_norm (same checkpoint); diag optional
    diag_wnorm = {}
    if diag_path and os.path.exists(diag_path):
        recs = _load_jsonl(diag_path)
        last = recs[-1]
        if last.get('output'):
            diag_wnorm['output.weight'] = last['output'].get('w_norm')

    print(f"=== WD-equilibrium ({os.path.basename(nud_path)}) eta={eta:.3e} lambda={wd} ===")
    print(f"Model: ||W_eq|| = ||U|| / sqrt(2*eta*lambda),  ||U|| = eta * ||update||\n")

    # aggregate per class: median update_norm, median current w_norm
    by_cls = {}
    for r in rows:
        cls = _class_of(r['name'])
        if cls in ('other',):
            continue
        un = r.get('update_norm')
        wn = r.get('w_norm')
        if un is None or wn is None:
            continue
        by_cls.setdefault(cls, {'update_norm': [], 'w_norm': []})
        by_cls[cls]['update_norm'].append(un)
        by_cls[cls]['w_norm'].append(wn)

    import statistics as st
    result = {'checkpoint': nud.get('checkpoint'), 'eta': eta, 'lambda': wd, 'classes': {}}
    hdr = f"  {'class':10} {'cur||W||':>9} {'||U||/step':>10}" + "".join(
        f" {'eq@'+str(m)+'x':>9}" for m in lambda_mults) + f"  {'lambda_to_pin_current':>22}"
    print(hdr)
    for cls, d in by_cls.items():
        cur_w = st.median(d['w_norm'])
        upd = st.median(d['update_norm'])
        U = eta * upd                      # per-step ||ΔW|| (tangential)
        eqs = {}
        for m in lambda_mults:
            lam = wd * m
            denom = 2 * eta * lam
            eqs[m] = (U / math.sqrt(denom)) if denom > 0 else float('inf')
        # what lambda would put equilibrium at the CURRENT norm?
        # ||W_eq||=cur_w => 2*eta*lam = (U/cur_w)^2 => lam = (U/cur_w)^2/(2*eta)
        lam_pin = ((U / cur_w) ** 2) / (2 * eta) if cur_w > 0 else float('inf')
        result['classes'][cls] = {
            'n': len(d['w_norm']), 'cur_w_norm': cur_w, 'U_per_step': U,
            'equilibrium_norm': {f'{m}x': eqs[m] for m in lambda_mults},
            'lambda_to_pin_current': lam_pin,
        }
        cells = "".join(f" {eqs[m]:>9.1f}" for m in lambda_mults)
        print(f"  {cls:10} {cur_w:>9.1f} {U:>10.4f}{cells}  {lam_pin:>22.4f}")

    print("\nINTERPRETATION:")
    print("  eq@1x  = equilibrium ||W|| at the CURRENT lambda. If >> cur||W||, the layer")
    print("           is still far below equilibrium -> will keep ramping a long time.")
    print("  eq@Nx  = equilibrium at N*lambda. Pick N so eq ~= a TARGET norm.")
    print("  lambda_to_pin_current = the WD that would make CURRENT norm the equilibrium")
    print("           (stop further growth here). Compare to current lambda to size the dial.")

    if out_path:
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=1)
        print(f"\nwrote {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nud", required=True, help="nud_*.json from normuon_update_decomp")
    ap.add_argument("--diag", default=None, help="diagnostics.jsonl (optional, for head w_norm)")
    ap.add_argument("--eta", type=float, required=True, help="current learning rate")
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--mults", default="1,2,4,10")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mults = tuple(float(x) for x in a.mults.split(","))
    run(a.nud, a.diag, a.eta, a.wd, lambda_mults=mults, out_path=a.out)


if __name__ == "__main__":
    main()
