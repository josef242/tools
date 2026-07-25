#!/usr/bin/env python3
"""Production-scale emission probe: run the SAME prefix-conditioned verbatim-overlap metric
used in the toy reps/capacity runs against wizard101 (4.6B) checkpoints.

Rook's probes:
  1. score the logit-registered alpha out-of-sample at 34.8x (cross-book class)
  2. settle the 2-4-dup acquittal at production scale (ASYMMETRIC: null is confounded --
     wizard101 saw only ~11% of the books corpus, so 2-4-dup lines had ~21-38% chance of
     ever being seen; a POSITIVE is informative, a NULL is not)
  3. forgetting curve across the step-10,750 data switch (raw books -> v8-derived)

Metric is byte-identical to emission_probe.py: prefix = first --prefix-tok tokens, continuation =
next --cont-tok; teacher-forced continuation NLL/token + greedy argmax verbatim overlap. Single
forward per probe (NOT autoregressive generation) -- this is why it is cheap even at 4.6B.

Loads via neo_common.load_model_and_tokenizer (the generate_neo.py path), bf16, optionally sharded
across GPUs. RoPE is left on AUTO: wizard101 checkpoints carry rope_fixed=True so AUTO selects
FIXED correctly -- do not override unless a checkpoint lacks that marker.

  python probe_wizard.py --ckpt <ckpt.pt|run_dir> --out reports/probe_wiz_<step>.json \
      [--probes bookclean/reports/emit_probes.jsonl] [--gpu 1] [--shard auto] [--max-memory 10GiB]
"""
import os, sys, json, argparse
import torch, torch.nn.functional as F
for _p in ('../common_fsdp2', '.'):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import neo_common as nc  # noqa: E402


def score_probe(model, enc, dev, text, ptok, ctok):
    """Identical metric to emission_probe.score_probe: (cont_nll_per_tok, greedy_verbatim_overlap)."""
    ids = [1] + enc.encode(text)
    if len(ids) < ptok + 4:
        return None
    pre, cont = ids[:ptok], ids[ptok:ptok + ctok]
    if not cont:
        return None
    seq = torch.tensor([pre + cont], device=dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(seq)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if logits.dim() == 3:
            logits = logits[0]
        lp = F.log_softmax(logits.float(), dim=-1)
        tgt = seq[0].to(lp.device)
        idxs = range(len(pre) - 1, len(pre) - 1 + len(cont))
        nll = -sum(lp[i, tgt[i + 1]].item() for i in idxs) / len(cont)
        hit = sum(1 for i in idxs if int(lp[i].argmax()) == int(tgt[i + 1]))
    return nll, hit / len(cont)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--probes', default='bookclean/reports/emit_probes.jsonl')
    ap.add_argument('--gpu', type=int, default=1, help='preferred GPU (default 1 = the 16GB card)')
    ap.add_argument('--shard', default='none', help="shard_strategy: none|auto (auto spreads across GPUs)")
    ap.add_argument('--max-memory', default=None, help='per-GPU cap for sharding, e.g. 10GiB')
    ap.add_argument('--prefix-tok', type=int, default=24)
    ap.add_argument('--cont-tok', type=int, default=48)
    a = ap.parse_args()
    a.ckpt = os.path.expanduser(a.ckpt)

    # neo_common expects device as a STRING (it does device.startswith("cuda"))
    dev_str = f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(dev_str)
    model, enc, cfg = nc.load_model_and_tokenizer(
        a.ckpt, device=dev_str, half_precision=True,      # bf16: 4.6B ~9.2GB
        shard_strategy=a.shard, preferred_gpu=a.gpu,
        max_memory_per_gpu=a.max_memory,
    )
    model.eval()
    # sharded models put inputs on the embedding's device
    in_dev = dev
    if hasattr(model, 'hf_device_map'):
        d0 = list(model.hf_device_map.values())[0]
        in_dev = torch.device(f'cuda:{d0}' if isinstance(d0, int) else d0)

    probes = [json.loads(l) for l in open(a.probes)]
    nlls, ovs, byclass = [], [], {}
    for p in probes:
        r = score_probe(model, enc, in_dev, p['text'], a.prefix_tok, a.cont_tok)
        if r is None:
            continue
        nll, ov = r
        nlls.append(nll); ovs.append(ov)
        byclass.setdefault(p.get('class', '?'), []).append((nll, ov))

    def m(x): return sum(x) / len(x) if x else None
    res = {'ckpt': a.ckpt, 'step': cfg.get('step') if isinstance(cfg, dict) else None,
           'n': len(nlls), 'cont_nll_per_tok': m(nlls), 'greedy_verbatim_overlap': m(ovs),
           'verbatim_frac_over_0.8': sum(1 for o in ovs if o > 0.8) / max(len(ovs), 1),
           'by_class': {c: {'n': len(v), 'cont_nll': m([x[0] for x in v]),
                            'overlap': m([x[1] for x in v])} for c, v in byclass.items()}}
    json.dump(res, open(a.out, 'w'), indent=1)
    print(f"[{os.path.basename(a.ckpt)}] n={res['n']} overlap={res['greedy_verbatim_overlap']:.4f} "
          f"cont_nll={res['cont_nll_per_tok']:.4f}")
    for c, d in res['by_class'].items():
        print(f"   {c:12s} n={d['n']:4d} overlap={d['overlap']:.4f} cont_nll={d['cont_nll']:.4f}")


if __name__ == '__main__':
    main()
