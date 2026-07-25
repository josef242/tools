#!/usr/bin/env python3
"""Metric #3: loss-spike / grad-norm anomaly counts from train logs (env-independent
log parse). Pre-registered prediction P4: arm A (v9, junk present) shows MORE
loss-spikes / grad-norm outliers than arm B (v11). Directly tests the "magnets cause
gradient anomalies" assumption.

Parses train_log.txt lines like:
  ... | st: 1 | ls: 10.562500 | ... | nrm: 3.2861 [2.0] | ...
Per run: robust grad-norm outlier count (nrm > median + K*MAD), loss-spike count
(ls rises beyond local noise), clipped-fraction (nrm > clip). Aggregates by arm.

Usage: python grad_anomaly.py <checkpoints_root_or_run_dirs...>
  e.g. python grad_anomaly.py /home/josef/brainbox/checkpoints/current/books-abl-*"""
import sys, os, re, glob, json
from statistics import median

LINE = re.compile(r'st:\s*(\d+).*?ls:\s*([\d.]+).*?nrm:\s*([\d.]+)\s*\[([\d.]+)\]')

def parse_run(run_dir):
    log = os.path.join(run_dir, 'train_log.txt')
    if not os.path.exists(log):
        return None
    steps = []
    for ln in open(log):
        m = LINE.search(ln)
        if m:
            steps.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    if len(steps) < 20:
        return None
    steps.sort()
    ls = [s[1] for s in steps]; nrm = [s[2] for s in steps]; clip = [s[3] for s in steps]
    # skip warmup (first 10%) for anomaly stats — early training is legitimately volatile
    w = max(10, len(steps) // 10)
    nrm_s = nrm[w:]; ls_s = ls[w:]
    med = median(nrm_s); mad = median([abs(x - med) for x in nrm_s]) or 1e-6
    grad_outliers = sum(1 for x in nrm_s if x > med + 6 * mad)        # robust >6 MAD
    clipped_frac = sum(1 for i in range(w, len(steps)) if nrm[i] > clip[i]) / max(len(steps) - w, 1)
    # loss-spike: ls rises by > 5x the median down-step magnitude
    downs = [ls_s[i - 1] - ls_s[i] for i in range(1, len(ls_s)) if ls_s[i] < ls_s[i - 1]]
    dmed = median(downs) if downs else 1e-6
    loss_spikes = sum(1 for i in range(1, len(ls_s)) if ls_s[i] - ls_s[i - 1] > 5 * dmed)
    return {'run': os.path.basename(run_dir.rstrip('/')), 'n_steps': len(steps),
            'grad_outliers': grad_outliers, 'loss_spikes': loss_spikes,
            'clipped_frac': round(clipped_frac, 3),
            'nrm_median': round(med, 3), 'final_ls': round(ls[-1], 4)}

def main():
    dirs = []
    for a in sys.argv[1:]:
        dirs += [d for d in glob.glob(a) if os.path.isdir(d)]
    if not dirs:
        print("no run dirs found"); return
    rows = [r for d in sorted(dirs) if (r := parse_run(d))]
    for r in rows:
        print(f"  {r['run']:<24} steps={r['n_steps']:<5} grad_outliers={r['grad_outliers']:<3} "
              f"loss_spikes={r['loss_spikes']:<3} clipped={r['clipped_frac']:<5} "
              f"nrm_med={r['nrm_median']} final_ls={r['final_ls']}")
    # aggregate by arm (v9 vs v11), excluding spike/probe
    def arm(name):
        if '-v9-' in name: return 'A_v9'
        if '-v11spike' in name: return 'C_v11spike'
        if '-v11-' in name: return 'B_v11'
        return 'other'
    agg = {}
    for r in rows:
        a = arm(r['run']); d = agg.setdefault(a, {'grad_outliers': [], 'loss_spikes': [], 'clipped_frac': []})
        d['grad_outliers'].append(r['grad_outliers']); d['loss_spikes'].append(r['loss_spikes'])
        d['clipped_frac'].append(r['clipped_frac'])
    print("\n=== METRIC #3 by arm (mean over seeds) — P4: A > B expected ===")
    for a in sorted(agg):
        d = agg[a]; n = len(d['grad_outliers'])
        print(f"  {a}: grad_outliers={sum(d['grad_outliers'])/n:.1f}  "
              f"loss_spikes={sum(d['loss_spikes'])/n:.1f}  clipped_frac={sum(d['clipped_frac'])/n:.3f}  (n={n})")
    json.dump({'runs': rows, 'by_arm': {a: {k: sum(v)/len(v) for k, v in d.items()}
              for a, d in agg.items()}}, open('bookclean/reports/grad_anomaly.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
