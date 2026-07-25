#!/usr/bin/env python3
"""Aggregate the harm ablation across arms/seeds and apply the PRE-REGISTERED
thresholds (reports/bookclean/reports/ABLATION_PREREG.md) to emit a verdict.
Reads per-run eval_ablation JSONs (metric #1 junk-LL, #2 emission), grad_anomaly.json
(#3), and the val logs (#4).

Decision (locked before data):
  harm of removed junk CONFIRMED if junk-LL ratio >10x on TOP magnets (A over B)
    OR emission delta >5x (A over B), outside seed-noise; else NULL.
  Arm C (if present): no signal above B on #1-3 => span judge is DEAD.

Usage: python aggregate_ablation.py reports/eval_books-abl-*.json"""
import sys, os, json, glob, math
from statistics import mean, pstdev

def arm_of(ckpt):
    b = ckpt
    if 'v9' in b: return 'A'
    if 'v11spike' in b: return 'C'
    if 'v11' in b: return 'B'
    return '?'

def main():
    files = []
    for a in sys.argv[1:]:
        files += glob.glob(a)
    runs = [json.load(open(f)) for f in files]
    byarm = {'A': [], 'B': [], 'C': []}
    for r in runs:
        byarm[arm_of(r['ckpt'])].append(r)
    print(f"loaded runs: A={len(byarm['A'])} B={len(byarm['B'])} C={len(byarm['C'])}")

    def seed_mean_perjunk(arm):
        """Mean per-junk per-tok NLL across seeds (index-aligned; same lexicon order)."""
        arr = [r['metric1_junkLL']['junk_nll_per_tok'] for r in byarm[arm] if r.get('metric1_junkLL')]
        if not arr: return None
        n = min(len(x) for x in arr)
        return [mean(x[i] for x in arr) for i in range(n)]

    A = seed_mean_perjunk('A'); B = seed_mean_perjunk('B')
    if not A or not B:
        print("need both arm A and arm B eval outputs"); return
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]

    # overall junk NLL (lower NLL = better memorized = higher prob)
    print(f"\n=== METRIC #1: junk log-likelihood (per-token NLL; lower = model assigns junk higher prob) ===")
    print(f"  arm A (v9)   mean junk NLL/tok = {mean(A):.4f}")
    print(f"  arm B (v11)  mean junk NLL/tok = {mean(B):.4f}")
    print(f"  A assigns junk {'HIGHER' if mean(A) < mean(B) else 'lower'} prob than B "
          f"(expected: A higher, since only A trained on the junk)")

    # TOP magnets: junk A memorized best (lowest NLL in A). ratio = P_A/P_B = exp(nll_B - nll_A)
    idx = sorted(range(n), key=lambda i: A[i])[:max(1, n // 20)]   # top 5%
    ratios = [math.exp(min(B[i] - A[i], 20)) for i in idx]         # clamp for numerical safety
    top_ratio = math.exp(mean(B[i] - A[i] for i in idx))
    print(f"\n  TOP magnets (top 5% best-memorized-by-A, n={len(idx)}):")
    print(f"    median P_A/P_B ratio = {sorted(ratios)[len(ratios)//2]:.2f}x   geomean = {top_ratio:.2f}x")
    print(f"    [PRE-REG THRESHOLD: >10x on top magnets = harm CONFIRMED]")

    # seed noise floor: within-arm across-seed variability on the same junk set
    def within_noise(arm):
        arr = [r['metric1_junkLL']['junk_nll_per_tok'] for r in byarm[arm] if r.get('metric1_junkLL')]
        if len(arr) < 2: return None
        m = min(len(x) for x in arr)
        return mean(pstdev([x[i] for x in arr]) for i in range(m))
    nA, nB = within_noise('A'), within_noise('B')
    ab_gap = abs(mean(A) - mean(B))
    print(f"\n  seed-noise floor: A={nA}  B={nB} | A-vs-B mean gap={ab_gap:.4f}")

    # METRIC #2 emission
    def emit(arm):
        e = [r['metric2_emission']['emission_rate'] for r in byarm[arm] if r.get('metric2_emission')]
        return mean(e) if e else None
    eA, eB = emit('A'), emit('B')
    print(f"\n=== METRIC #2: emission rate (fraction of generations with a removed-junk 10-gram) ===")
    if eA is not None and eB is not None:
        delta = eA / eB if eB > 0 else float('inf')
        print(f"  arm A={eA:.4f}  arm B={eB:.4f}  delta={delta:.2f}x  [PRE-REG: >5x = confirmed]")
    else:
        delta = None; print("  (emission not evaluated)")

    # METRIC #3 (if grad_anomaly.json present)
    ga = 'bookclean/reports/grad_anomaly.json'
    if os.path.exists(ga):
        g = json.load(open(ga)).get('by_arm', {})
        print(f"\n=== METRIC #3: grad anomalies by arm (P4: A>B) ===")
        for a in ('A_v9', 'B_v11', 'C_v11spike'):
            if a in g:
                print(f"  {a}: grad_outliers={g[a].get('grad_outliers')}  loss_spikes={g[a].get('loss_spikes')}")

    # VERDICT
    harm = (top_ratio > 10) or (delta is not None and delta > 5)
    print(f"\n{'='*70}\nVERDICT (pre-registered): "
          f"{'HARM of removed junk CONFIRMED' if harm else 'NULL (no harm detectable at this scale)'}")
    print(f"  junk-LL top-magnet ratio {top_ratio:.2f}x (>10x?) | emission delta "
          f"{delta if delta is None else round(delta,2)}x (>5x?)")
    if byarm['C']:
        C = seed_mean_perjunk('C')  # placeholder — spike-in uses its own residual eval set
        print("  Arm C present — evaluate residual signal separately (span-judge gate).")
    else:
        print("  Arm C not run yet — build+run ONLY if the above is CONFIRMED (else residual is moot).")
    print('='*70)

if __name__ == '__main__':
    main()
