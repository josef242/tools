"""Spectral contribution audit (Math Agent's mechanism-closer) — REAL CE gradient.

Proves WHY Newton-Schulz turns a radial-null CE gradient into an anti-radial update.
Mechanism (Math Agent): with G = U diag(sigma) V^T and a_i = u_i^T W v_i,
  raw radial dot   <G,W>      = sum_i sigma_i a_i   (sigma-WEIGHTED)  ~ 0
  post-NS dot      <UV^T,W>   = sum_i a_i           (UNWEIGHTED)      < 0
NS flattens sigma, so the unweighted sum dominates -> anti-radial update from null grad.

For each body matrix (REAL CE gradient on a real batch, single-card full matrices):
  1. SVD of G; compute a_i = u_i^T W v_i.
  2. Bin singular values into quantiles; report sum(sigma_i a_i) vs sum(a_i) per bin + total.
  3. cos(stage, W) after EACH Newton-Schulz iteration (raw -> NS1..NS5), expect monotonic
     climb toward ~-0.0128 (closes the singular-value-flattening story).

Run: python ns_spectral_audit.py --ckpt <pt> --config <yaml> [--n 12] [--seq 1024]
"""
import os, sys, math, argparse, json, statistics as st
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("../common_fsdp2", "../saved_code"):
    _ap = os.path.normpath(os.path.join(_HERE, _p))
    if _ap not in sys.path:
        sys.path.insert(0, _ap)
import logger  # noqa: E402
logger._instance.set_logdir("./logs"); logger._instance.set_rank(0)
import neo_common as nc  # noqa: E402
from zloss_row_center_probe import resolve_ckpt, _resolve_own_groups  # noqa: E402
from rare_token_nll_probe import build_panel  # noqa: E402
from muon_fsdp2 import nsloop_torch  # noqa: E402


def _cos(a, b):
    a = a.float(); b = b.float()
    d = (a * b).sum().item(); an = a.norm().item(); bn = b.norm().item()
    return (d / (an * bn)) if (an > 0 and bn > 0) else 0.0


def _isbody(n):
    return any(n.endswith(s) for s in ('wo.weight', 'w2.weight', 'wq.weight',
                                       'wk.weight', 'wv.weight', 'w1.weight', 'w3.weight'))


def _ns_trace(G, W, steps):
    """cos(X,W) after each NS iteration, replicating zeropower_via_newtonschulz5 stage by stage."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = False
    if G.size(-2) > G.size(-1):
        X = X.mT; transposed = True
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    cos_by_iter = []
    Wt = (W.mT if transposed else W).float()
    for _ in range(steps):
        X = nsloop_torch(X, 1, a=a, b=b, c=c)  # one iteration
        cos_by_iter.append(_cos(X.float(), Wt))
    return cos_by_iter


def run(ckpt, groups_override=None, config_path=None, n=12, seq=1024, nbins=5, seed=0):
    dev = nc.detect_device(None)
    path = resolve_ckpt(ckpt)
    logger.print_and_log(f"=== NS spectral audit: {os.path.basename(path)} on {dev} ===")
    model, enc, cfg = nc.load_model_and_tokenizer(
        path, device=dev, half_precision=True, shard_strategy="none", use_keel=None)
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw.train()
    pad_id = int(getattr(cfg, "pad_id", 0) or 0)
    data_root = getattr(cfg, "data_root_path", "../../notebooks/datasets/tokenized/llama/")
    groups = groups_override or _resolve_own_groups(cfg, path, config_path=config_path)[0]
    win = min(seq, int(getattr(getattr(model, "params", None), "max_seq_len", seq) or seq))
    tokens, targets = build_panel(data_root, groups, win + 1, dev, seed=seed)
    x = tokens[:, :win]; y = targets[:win]

    # real CE gradient (train-branch fused CCE — same path as production)
    for p in raw.parameters():
        p.grad = None
    out = raw(x, y, active_layers=None, scaffold_mode=False)
    loss = out[1] if isinstance(out, (tuple, list)) else out
    loss.backward()

    bodies = [(nm, p) for nm, p in raw.named_parameters() if _isbody(nm) and p.dim() == 2 and p.grad is not None]
    logger.print_and_log(f"loss={loss.item():.4f}; {len(bodies)} body matrices, auditing {min(n,len(bodies))}")

    agg = {'raw_dot': [], 'unweighted_dot': [], 'cos_rawG': [], 'cos_polar': [],
           'bin_weighted': [[] for _ in range(nbins)], 'bin_unweighted': [[] for _ in range(nbins)],
           'ns_iter_cos': []}
    for nm, p in bodies[:n]:
        W = p.detach().float()
        G = p.grad.detach().float()
        # SVD of the real gradient
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        V = Vh.mH
        # a_i = u_i^T W v_i  (diagonal of U^T W V)
        a = torch.einsum('mi,mn,ni->i', U, W, V)   # [k]
        raw_dot = (S * a).sum().item()             # = <G,W>  (sigma-weighted)
        unw_dot = a.sum().item()                   # = <UV^T,W> (unweighted, polar)
        agg['raw_dot'].append(raw_dot)
        agg['unweighted_dot'].append(unw_dot)
        agg['cos_rawG'].append(_cos(G, W))
        # cos(polar, W) where polar = U V^T
        polar = U @ V.mT
        agg['cos_polar'].append(_cos(polar, W))
        # bin by singular value (quantiles), report weighted vs unweighted contribution
        k = S.numel()
        order = torch.argsort(S, descending=True)   # high sigma first
        for bi in range(nbins):
            lo = bi * k // nbins; hi = (bi + 1) * k // nbins
            idx = order[lo:hi]
            agg['bin_weighted'][bi].append((S[idx] * a[idx]).sum().item())
            agg['bin_unweighted'][bi].append(a[idx].sum().item())
        # per-NS-iteration cos trace
        agg['ns_iter_cos'].append(_ns_trace(G, W, 5))

    def med(v): return st.median(v) if v else 0.0
    logger.print_and_log("\n=== per-matrix radial dots (median over audited matrices) ===")
    logger.print_and_log(f"  <G,W> sigma-WEIGHTED  (raw grad radial)   : {med(agg['raw_dot']):+.4e}  (expect ~0)")
    logger.print_and_log(f"  <UV^T,W> UNWEIGHTED   (polar/NS radial)    : {med(agg['unweighted_dot']):+.4e}  (expect <0)")
    logger.print_and_log(f"  cos(raw G, W)   : {med(agg['cos_rawG']):+.5f}  (expect ~0)")
    logger.print_and_log(f"  cos(polar, W)   : {med(agg['cos_polar']):+.5f}  (expect <0, ~ -0.013)")
    logger.print_and_log("\n=== contribution by singular-value bin (Q0=highest sigma): weighted | unweighted ===")
    for bi in range(nbins):
        logger.print_and_log(f"  Q{bi}: weighted={med(agg['bin_weighted'][bi]):+.4e}   unweighted={med(agg['bin_unweighted'][bi]):+.4e}")
    # per-NS-iteration cos (median across matrices, per iter)
    niter = len(agg['ns_iter_cos'][0]) if agg['ns_iter_cos'] else 0
    iters = [med([row[j] for row in agg['ns_iter_cos']]) for j in range(niter)]
    logger.print_and_log("\n=== cos(X,W) after each NS iteration (median) — expect monotonic climb to ~-0.0128 ===")
    logger.print_and_log("  raw=0  " + "  ".join(f"NS{j+1}={iters[j]:+.5f}" for j in range(niter)))
    logger.print_and_log(
        "\nRead: if WEIGHTED total ~0 but UNWEIGHTED total <0 (high-sigma bins cancel low-sigma "
        "in weighted, but unweighted goes negative) => MECHANISM CLOSED: NS spectral flattening "
        "converts radial-null gradient into anti-radial update.")
    return {'raw_dot_med': med(agg['raw_dot']), 'unweighted_dot_med': med(agg['unweighted_dot']),
            'cos_rawG_med': med(agg['cos_rawG']), 'cos_polar_med': med(agg['cos_polar']),
            'bin_weighted': [med(b) for b in agg['bin_weighted']],
            'bin_unweighted': [med(b) for b in agg['bin_unweighted']],
            'ns_iter_cos': iters}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--nbins", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    go = [g.strip() for g in a.groups.split(",")] if a.groups else None
    res = run(a.ckpt, groups_override=go, config_path=a.config, n=a.n, seq=a.seq, nbins=a.nbins, seed=a.seed)
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)
        logger.print_and_log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
