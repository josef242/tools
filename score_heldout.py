#!/usr/bin/env python3
"""Score a checkpoint's mean NLL/tok on the held-out clean-val span set (build_heldout_val.py).
Reuses eval_ablation's loader + scorer so numbers are directly comparable to the junk/control metrics.
Run on both arms' final checkpoints; the A-B gap is the opportunity-cost/displacement measure.

  python score_heldout.py --ckpt <run_or_ckpt> --out reports/heldout_<run>.json [--gpu 0]
"""
import os, sys, json, argparse
for _p in ('../common_fsdp2', '.'):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from eval_ablation import load, encode_spans, batched_nll  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--spans', default='bookclean/reports/heldout_val.jsonl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--max-seq-tok', type=int, default=256)
    a = ap.parse_args()
    a.ckpt = os.path.expanduser(a.ckpt)
    model, enc, dev = load(a.ckpt, a.gpu)
    txt = [json.loads(l)['text'] for l in open(a.spans)]
    seqs = encode_spans(enc, txt, a.max_seq_tok)
    r = batched_nll(model, seqs, dev)
    pt = [n / max(t, 1) for n, t in r]
    mean = sum(pt) / len(pt) if pt else None
    json.dump({'ckpt': a.ckpt, 'n': len(pt), 'heldout_nll_per_tok': mean}, open(a.out, 'w'), indent=1)
    print(f"[{os.path.basename(a.ckpt.rstrip('/'))}] held-out clean-val NLL/tok = {mean:.4f}  (n={len(pt)})")

if __name__ == '__main__':
    main()
