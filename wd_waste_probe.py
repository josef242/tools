"""WD-gradient-waste probe — Part A (Nexus #167).

Tests Rook's hypothesis that body layers (not just the output head) accumulate
loss-null "gauge"-like components that decoupled weight decay then wastes gradient
fighting — the suspected source of the ever-growing avg gradient norm.

THE INSIGHT: wherever the loss is invariant to a weight transform, the
loss-gradient is 0 in that direction, the optimizer drifts there unchecked, and
decoupled WD (-lambda*W, purely radial) generates gradient fighting whatever
accumulated = wasted motion.
  - HEAD: softmax shift-invariance -> loss-null = common-mode row-mean mu.
  - BODY (pre-norm matrices): RMSNorm SCALE-invariance, L(cW)=L(W) ->
    <g_loss, W> = 0, so the loss-grad is orthogonal to the radial direction, but
    -lambda*W is ENTIRELY radial -> WD gradient is ~loss-null for pre-norm W.

PART A is nearly free: the decoupled-WD gradient magnitude is exactly
||lambda*W|| = lambda*||W||_F, and per-layer w_norm + g_norm (CE-only, captured
pre-WD) are ALREADY in diagnostics.jsonl for every diag step across the run. So
this is a pure LOG PARSE — no GPU, no checkpoint loads. Computes per layer:
  wd_grad_mag   = lambda * w_norm
  wd_grad_share = wd_grad_mag / (wd_grad_mag + g_norm)   # WD's fraction of total motion on that weight
  loss_grad     = g_norm                                  # the real signal
and tracks ||W||_F, wd_grad_mag, summed wd_grad_mag, and shares over training.

Confirmed prerequisites (muon_fsdp2 / train_mara): WD is DECOUPLED
(p.mul_(1-eta*lambda)), and diagnostics g_norm is the LOSS gradient captured
AFTER backward, BEFORE the optimizer's WD -> g_norm excludes the -lambda*W term,
so the decomposition is clean.

Run:
    python wd_waste_probe.py --diag <diagnostics.jsonl> [--wd 0.02] [--out report.json]
"""
import os
import sys
import json
import argparse


def _block_stats(block, wd):
    """(w_norm, g_norm, wd_grad_mag, wd_grad_share) for a diag block, or None."""
    if not block:
        return None
    wn = block.get('w_norm')
    gn = block.get('g_norm')
    if wn is None or gn is None:
        return None
    wd_mag = wd * wn
    denom = wd_mag + gn
    share = (wd_mag / denom) if denom > 1e-12 else 0.0
    return {'w_norm': wn, 'g_norm': gn, 'wd_grad_mag': wd_mag, 'wd_grad_share': share}


def _load_jsonl(path):
    """Tolerant JSONL reader: handles clean one-object-per-line files AND older
    files with multiple concatenated objects on a line (uses raw_decode to walk
    each line), skipping any unparseable fragments."""
    recs = []
    dec = json.JSONDecoder()
    for line in open(path):
        s = line.strip()
        while s:
            try:
                obj, end = dec.raw_decode(s)
            except json.JSONDecodeError:
                break
            recs.append(obj)
            s = s[end:].lstrip()
    return recs


def analyze(diag_path, wd=0.02):
    recs = _load_jsonl(diag_path)
    series = []
    for r in recs:
        step = r.get('step')
        layers = r.get('layers') or []
        per_layer = {}
        summed_wd = 0.0
        summed_g = 0.0
        # output head (common-mode gauge, not norm-scale — but the WD term
        # lambda*||W|| applies identically; it's the positive control)
        out = _block_stats(r.get('output'), wd)
        if out:
            per_layer['output'] = out
            summed_wd += out['wd_grad_mag']
            summed_g += out['g_norm']
        # body: attn + ffn per layer (pre-norm matrices -> scale-invariant)
        for ls in layers:
            idx = ls.get('idx')
            for sub in ('attn', 'ffn'):
                st = _block_stats(ls.get(sub), wd)
                if st:
                    per_layer[f'L{idx}.{sub}'] = st
                    summed_wd += st['wd_grad_mag']
                    summed_g += st['g_norm']
        # tok_embeddings (no downstream norm on its output in the usual sense, but
        # track it — embeddings feed the first norm, so arguably scale-relevant)
        emb = _block_stats(r.get('tok_embeddings'), wd)
        if emb:
            per_layer['tok_embeddings'] = emb
            summed_wd += emb['wd_grad_mag']
            summed_g += emb['g_norm']

        total_motion = summed_wd + summed_g
        series.append({
            'step': step,
            'summed_wd_grad_mag': summed_wd,
            'summed_loss_grad': summed_g,
            'summed_wd_share': (summed_wd / total_motion) if total_motion > 1e-12 else 0.0,
            'per_layer': per_layer,
        })
    return series


def _depth_buckets(per_layer):
    """Mean wd_grad_share by depth third (early/mid/late body layers)."""
    body = [(k, v) for k, v in per_layer.items() if k.startswith('L')]
    if not body:
        return {}
    def li(k):
        return int(k[1:].split('.')[0])
    body.sort(key=lambda kv: li(kv[0]))
    n = len(body)
    out = {}
    for name, lo, hi in [('early', 0, n // 3), ('mid', n // 3, 2 * n // 3), ('late', 2 * n // 3, n)]:
        chunk = body[lo:hi]
        if chunk:
            out[name] = sum(v['wd_grad_share'] for _, v in chunk) / len(chunk)
    return out


def norm_growth(series, recent_frac=0.25):
    """Per-matrix ||W|| growth analysis — THE test for benign-equilibrium vs
    pathological ramping. For each tracked matrix returns:
      w_first, w_last, growth_pct (last/first - 1),
      recent_slope_per_kstep: linear ||W|| slope over the last `recent_frac` of
        the run, normalized to per-1000-steps and as a %/kstep of current ||W||
        — a CURRENTLY-flat norm => equilibrium even if it grew early.
    Flat recent slope = equilibrium; persistent positive slope = still ramping."""
    if len(series) < 2:
        return {}
    keys = set()
    for s in series:
        keys.update(s['per_layer'].keys())
    out = {}
    n = len(series)
    r0 = max(0, int(n * (1 - recent_frac)))
    recent = series[r0:]
    for k in keys:
        pts = [(s['step'], s['per_layer'][k]['w_norm']) for s in series if k in s['per_layer']]
        if len(pts) < 2:
            continue
        w_first, w_last = pts[0][1], pts[-1][1]
        rpts = [(st, w) for st, w in pts if st >= recent[0]['step']]
        slope_per_k = 0.0
        if len(rpts) >= 2:
            # simple least-squares slope of w vs step, scaled to per-1000-steps
            xs = [p[0] for p in rpts]; ys = [p[1] for p in rpts]
            mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = sum((x - mx) ** 2 for x in xs)
            slope_per_k = (num / den * 1000.0) if den > 0 else 0.0
        out[k] = {
            'w_first': w_first, 'w_last': w_last,
            'growth_pct': (w_last / w_first - 1.0) if w_first > 0 else 0.0,
            'recent_slope_per_kstep': slope_per_k,
            'recent_slope_pct_per_kstep': (slope_per_k / w_last) if w_last > 0 else 0.0,
        }
    return out


def report(series, wd):
    if not series:
        print("no diagnostic records found")
        return
    first, last = series[0], series[-1]
    print(f"=== WD-waste Part A (lambda={wd}) — {len(series)} diag points, "
          f"steps {first['step']}..{last['step']} ===\n")

    # Summed trajectory (the headline: does total WD-share grow?)
    print("SUMMED over all tracked matrices:")
    print(f"  {'step':>7} {'sum_wd_grad':>12} {'sum_loss_grad':>13} {'wd_share':>9}")
    stride = max(1, len(series) // 12)
    for s in series[::stride] + [last]:
        print(f"  {s['step']:>7} {s['summed_wd_grad_mag']:>12.3f} "
              f"{s['summed_loss_grad']:>13.3f} {s['summed_wd_share']:>8.1%}")

    # Output head trajectory (positive control — known gauge accumulator)
    print("\nOUTPUT HEAD (positive control):")
    print(f"  {'step':>7} {'w_norm':>9} {'wd_grad':>9} {'loss_grad':>9} {'wd_share':>9}")
    for s in series[::stride] + [last]:
        o = s['per_layer'].get('output')
        if o:
            print(f"  {s['step']:>7} {o['w_norm']:>9.1f} {o['wd_grad_mag']:>9.3f} "
                  f"{o['g_norm']:>9.3f} {o['wd_grad_share']:>8.1%}")

    # Body depth breakdown at first vs last (is it head-only or body too? where?)
    print("\nBODY wd_grad_share by depth (mean over attn+ffn in each third):")
    fb, lb = _depth_buckets(first['per_layer']), _depth_buckets(last['per_layer'])
    for k in ('early', 'mid', 'late'):
        f = fb.get(k); l = lb.get(k)
        if f is not None and l is not None:
            print(f"  {k:>6}: {f:6.1%} (@{first['step']})  ->  {l:6.1%} (@{last['step']})   d={l-f:+.1%}")

    # Worst individual body layers at the last point
    print("\nTOP-8 body matrices by wd_grad_share (last point):")
    body = [(k, v) for k, v in last['per_layer'].items() if k.startswith('L')]
    body.sort(key=lambda kv: kv[1]['wd_grad_share'], reverse=True)
    for k, v in body[:8]:
        print(f"  {k:>12}: wd_share={v['wd_grad_share']:6.1%}  "
              f"||W||={v['w_norm']:8.1f}  wd_grad={v['wd_grad_mag']:.3f}  loss_grad={v['g_norm']:.4f}")

    # === THE RAMPING TEST: ||W|| growth, full-run and RECENT slope ===
    # benign equilibrium = flat (recent slope ~0); pathological = still climbing.
    ng = norm_growth(series)
    body_ng = {k: v for k, v in ng.items() if k.startswith('L')}
    print("\n=== ||W|| RAMPING TEST (equilibrium vs still-climbing) ===")
    # Body summary: how many matrices are essentially flat recently?
    if body_ng:
        slopes = [v['recent_slope_pct_per_kstep'] for v in body_ng.values()]
        flat = sum(1 for s in slopes if abs(s) < 0.005)   # <0.5%/kstep = flat
        climbing = sum(1 for s in slopes if s >= 0.005)
        print(f"  BODY ({len(body_ng)} matrices): {flat} flat (<0.5%/kstep), "
              f"{climbing} still climbing (>=0.5%/kstep). "
              f"median recent slope {sorted(slopes)[len(slopes)//2]*100:+.3f}%/kstep")
        print("  Top-8 body matrices by RECENT ||W|| slope (the ones still ramping, if any):")
        for k, v in sorted(body_ng.items(), key=lambda kv: kv[1]['recent_slope_pct_per_kstep'], reverse=True)[:8]:
            print(f"    {k:>12}: ||W|| {v['w_first']:7.1f} -> {v['w_last']:7.1f} "
                  f"(full {v['growth_pct']:+6.1%})  recent {v['recent_slope_pct_per_kstep']*100:+.3f}%/kstep")
    # Head explicitly (the known accumulator)
    if 'output' in ng:
        o = ng['output']
        print(f"  OUTPUT HEAD: ||W|| {o['w_first']:.1f} -> {o['w_last']:.1f} "
              f"(full {o['growth_pct']:+.1%})  recent {o['recent_slope_pct_per_kstep']*100:+.3f}%/kstep")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True, help="path to diagnostics.jsonl")
    ap.add_argument("--wd", type=float, default=0.02, help="weight_decay lambda (default 0.02)")
    ap.add_argument("--out", default=None, help="optional JSON dump of the full series")
    a = ap.parse_args()
    series = analyze(a.diag, wd=a.wd)
    report(series, a.wd)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(series, f, indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
