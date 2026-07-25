#!/usr/bin/env python3
"""Emission probe (Rook #221): the deployment-relevant emission mode. Unconditional
sampling is ~0 by slot-competition (and slow on Pascal), so instead we PROMPT with
junk-context prefixes -- feed the OPENING of a copyright/CIP page and measure whether
the model continues it VERBATIM into the memorized boilerplate. Per-class, per-checkpoint.

For each probe span (a cross-book boilerplate span from the harm class), split it at a
prefix boundary; teacher-force nothing -- GENERATE from the prefix and score how much of
the true continuation the model reproduces:
  - exact-continuation NLL/tok (low = the model 'knows' the rest = emission risk), and
  - greedy verbatim overlap: fraction of the true continuation greedy-decoded correctly.
Compare arm A (saw the junk) vs arm B (didn't). Pre-reg P4: A emits, B doesn't; furniture ~0 both.

Usage:
  python emission_probe.py --ckpt <run_dir> --out reports/emit_<run>.json \
      [--probes bookclean/reports/emit_probes.jsonl] [--gpu 0] [--prefix-tok 24 --cont-tok 48]
Build the probe set first with build_emit_probes.py (cross-book class, len>=~80 tok)."""
import os, sys, json, argparse, re
import torch, torch.nn.functional as F
for _p in ('../common_fsdp2', '.'):
    if _p not in sys.path: sys.path.insert(0, _p)
import neo_common as nc
from eval_ablation import resolve_ckpt, load, _BF16_OK  # reuse loader (SDPA-fix + fp32 for Pascal)
from contextlib import nullcontext

def score_probe(model, enc, dev, text, ptok, ctok):
    """Prefix = first ptok tokens; continuation = next ctok. Return (cont_nll_per_tok, greedy_overlap)."""
    ids = [1] + enc.encode(text)                      # BOS + span
    if len(ids) < ptok + 4:
        return None
    pre = ids[:ptok]; cont = ids[ptok:ptok + ctok]
    if not cont:
        return None
    # (1) teacher-forced NLL of the TRUE continuation given the prefix
    seq = torch.tensor([pre + cont], device=dev)
    actx = torch.autocast('cuda', dtype=torch.bfloat16) if _BF16_OK else nullcontext()
    with torch.no_grad(), actx:
        logits = model(seq)[0][0]                     # (T,V)
        lp = F.log_softmax(logits.float(), dim=-1)
        idxs = range(len(pre) - 1, len(pre) - 1 + len(cont))
        nll = -sum(lp[i, seq[0, i + 1]].item() for i in idxs) / len(cont)
        # (2) greedy verbatim overlap: does argmax reproduce the true continuation?
        hit = sum(1 for i in idxs if int(lp[i].argmax()) == int(seq[0, i + 1]))
        overlap = hit / len(cont)
    return nll, overlap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--probes', default='bookclean/reports/emit_probes.jsonl')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--prefix-tok', type=int, default=24); ap.add_argument('--cont-tok', type=int, default=48)
    a = ap.parse_args()
    a.ckpt = os.path.expanduser(a.ckpt)
    model, enc, dev = load(a.ckpt, a.gpu)
    probes = [json.loads(l) for l in open(a.probes)]
    nlls = []; overlaps = []; byclass = {}
    for p in probes:
        r = score_probe(model, enc, dev, p['text'], a.prefix_tok, a.cont_tok)
        if r is None: continue
        nll, ov = r; nlls.append(nll); overlaps.append(ov)
        c = p.get('class', '?'); byclass.setdefault(c, []).append((nll, ov))
    def m(x): return sum(x) / len(x) if x else None
    res = {'ckpt': a.ckpt, 'n': len(nlls),
           'cont_nll_per_tok': m(nlls), 'greedy_verbatim_overlap': m(overlaps),
           'verbatim_frac_over_0.8': sum(1 for o in overlaps if o > 0.8) / max(len(overlaps), 1),
           'by_class': {c: {'n': len(v), 'cont_nll': m([x[0] for x in v]),
                            'overlap': m([x[1] for x in v])} for c, v in byclass.items()}}
    json.dump(res, open(a.out, 'w'), indent=1)
    print(f"[{os.path.basename(a.ckpt.rstrip('/'))}] cont_nll/tok={res['cont_nll_per_tok']:.3f} "
          f"verbatim_overlap={res['greedy_verbatim_overlap']:.3f} "
          f"frac>0.8={res['verbatim_frac_over_0.8']:.3f}")

if __name__ == '__main__':
    main()
