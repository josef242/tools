#!/usr/bin/env python3
"""Harm-ablation eval harness — metrics #1 (junk log-likelihood) and #2 (emission).
Runs on rig-31 (single GPU, no torchrun) against a trained checkpoint dir.

Metric #1 (primary, zero sampling noise): teacher-force the removed-junk sequences
(the v9->v11 diff = a free labeled junk lexicon) AND a matched content control through
the model; report per-sequence NLL. Arm A (v9, saw the junk) should assign LOWER NLL
(higher prob) to junk than arm B (v11, junk removed). The control (content present in
BOTH arms) should show NO A-vs-B gap -> isolates that the junk gap is the removal.
Pre-registered threshold: junk-LL ratio >10x on TOP magnets (A over B) = harm confirmed.

Metric #2: fraction of generations containing a removed-junk 10-gram (unconditional +
prompted from pristine prefixes). Threshold: emission delta >5x (A over B) = confirmed.

Usage:
  python eval_ablation.py --ckpt <run_dir> --out reports/eval_<run>.json \
      [--junk reports/junk_lexicon.jsonl] [--control reports/content_control.jsonl] \
      [--max-seq-tok 256] [--emit 512 --emit-tok 256]
Then aggregate across arms with aggregate_ablation.py."""
import os, sys, json, argparse, re
import torch
import torch.nn.functional as F
# neo_common + the model live in ../common_fsdp2 (NOT common / common_fsdp1) — must be
# on sys.path, prioritized, before the import.
for _p in ('../common_fsdp2', '.'):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import neo_common as nc

TOK_PATH = "../tokenizers/llama_tokenizer"
SPECIAL  = "../../notebooks/datasets/tokenized/llama/tokenizer_config.json"

def resolve_ckpt(path):
    """A dir -> its latest model_<step>.pt (load_model_and_tokenizer wants a FILE)."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        pts = []
        for f in os.listdir(path):
            if f.startswith('model_') and f.endswith('.pt'):
                m = re.search(r'_(\d+)\.pt', f)
                if m:
                    pts.append((int(m.group(1)), os.path.join(path, f)))
        if not pts:
            raise FileNotFoundError(f"no model_*.pt in {path}")
        pts.sort(reverse=True)
        print(f"[ckpt] selected {os.path.basename(pts[0][1])} (step {pts[0][0]})")
        return pts[0][1]
    raise FileNotFoundError(path)

def load(ckpt, gpu):
    ckpt = resolve_ckpt(ckpt)
    dev = nc.detect_device(preferred_gpu=gpu)
    model, enc, cfg = nc.load_model_and_tokenizer(
        ckpt, device=dev, tok_kind="llama", tok_path=TOK_PATH,
        special_tokens=SPECIAL, use_keel=None, qk_norm_mode="before_rope")
    model.eval()
    # FlexAttention doesn't run on Pascal (rig-27 1070s) / torch<2.9. Force the SDPA
    # causal path by disabling the flex-mask triggers. VALID HERE: eval spans are SHORT
    # single-doc sequences (one BOS at pos 0, len << 512 SWA window), so doc-mask, SWA,
    # and doc-pos-reset are provable no-ops -> byte-identical log-probs (per the model's
    # own no-BOS/short-seq parity notes). On flex-capable GPUs this changes nothing.
    for _m in (getattr(model, '_orig_mod', None), model):
        p = getattr(_m, 'params', None)
        if p is not None:
            for attr in ('doc_attn_mask', 'swa_enabled', 'doc_pos_reset'):
                if hasattr(p, attr):
                    setattr(p, attr, False)
    return model, enc, dev

def encode_spans(enc, spans, max_tok):
    """Encode each text span to llama ids with a leading BOS (id=1, matching training)."""
    out = []
    bos = 1
    for s in spans:
        ids = enc.encode(s)
        if not ids:
            continue
        ids = [bos] + ids[:max_tok]
        out.append(ids)
    return out

# Pascal (sm_6x) has no bf16 tensor cores -> autocast bf16 is EMULATED and slow.
# Native fp32 is faster there; use bf16 autocast only on Ampere+ (cc>=8).
_BF16_OK = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

def batched_nll(model, seqs, dev, batch=16):
    """Per-sequence: (summed_nll, n_target_tokens). Uses get_batch_nll semantics."""
    from contextlib import nullcontext
    res = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        maxlen = max(len(s) for s in chunk)
        toks = torch.zeros(len(chunk), maxlen, dtype=torch.long)
        mask = torch.zeros(len(chunk), maxlen, dtype=torch.long)
        for j, s in enumerate(chunk):
            toks[j, :len(s)] = torch.tensor(s)
            mask[j, :len(s)] = 1
        toks = toks.to(dev); mask = mask.to(dev)
        actx = torch.autocast("cuda", dtype=torch.bfloat16) if _BF16_OK else nullcontext()
        with torch.no_grad(), actx:
            logits = model(toks)[0]
            odev = logits.device
            tsh = toks[..., 1:].to(odev); msh = mask[..., 1:].to(odev)
            loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, logits.size(-1)),
                                   tsh.reshape(-1), reduction='none').view(toks.size(0), -1)
            snll = (loss * msh).sum(dim=1).float().cpu()
            ntok = msh.sum(dim=1).float().cpu()
        for k in range(len(chunk)):
            res.append((float(snll[k]), int(ntok[k])))
    return res

def metric1(model, enc, dev, junk_path, control_path, max_tok):
    junk_txt = [json.loads(l)['text'] for l in open(junk_path)]
    ctrl_txt = [json.loads(l)['text'] for l in open(control_path)] if os.path.exists(control_path) else []
    js = encode_spans(enc, junk_txt, max_tok)
    cs = encode_spans(enc, ctrl_txt, max_tok)
    jr = batched_nll(model, js, dev)
    cr = batched_nll(model, cs, dev) if cs else []
    # per-token NLL per sequence (comparable across lengths)
    j_pt = [n / max(t, 1) for n, t in jr]
    c_pt = [n / max(t, 1) for n, t in cr]
    def mean(x): return sum(x) / len(x) if x else None
    return {
        'junk_mean_nll_per_tok': mean(j_pt), 'junk_n': len(j_pt),
        'ctrl_mean_nll_per_tok': mean(c_pt), 'ctrl_n': len(c_pt),
        # keep per-sequence junk NLLs so aggregate can pick TOP magnets (lowest NLL = best memorized)
        'junk_nll_per_tok': j_pt,
    }

def build_junk_ngrams(junk_path, k=10):
    grams = set()
    for l in open(junk_path):
        w = re.sub(r'\d', '#', json.loads(l)['text'].lower()).split()
        for i in range(len(w) - k + 1):
            grams.add(' '.join(w[i:i + k]))
    return grams

def metric2(model, enc, dev, junk_path, n_emit, emit_tok):
    grams = build_junk_ngrams(junk_path)
    hits = 0; done = 0
    bos = torch.tensor([[1]], dtype=torch.long, device=dev)
    from contextlib import nullcontext
    for _ in range(n_emit):
        actx = torch.autocast("cuda", dtype=torch.bfloat16) if _BF16_OK else nullcontext()
        with torch.no_grad(), actx:
            cur = bos.clone()
            for _ in range(emit_tok):
                logits = model(cur)[0][:, -1, :]
                probs = F.softmax(logits.float() / 0.8, dim=-1)
                nxt = torch.multinomial(probs, 1)
                cur = torch.cat([cur, nxt], dim=1)
        txt = enc.decode(cur[0].tolist())
        w = re.sub(r'\d', '#', txt.lower()).split()
        found = any(' '.join(w[i:i + 10]) in grams for i in range(max(0, len(w) - 9)))
        hits += int(found); done += 1
    return {'emit_samples': done, 'emit_junk_hits': hits,
            'emission_rate': hits / max(done, 1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--junk', default='bookclean/reports/junk_lexicon.jsonl')
    ap.add_argument('--control', default='bookclean/reports/content_control.jsonl')
    ap.add_argument('--max-seq-tok', type=int, default=256)
    ap.add_argument('--emit', type=int, default=256)
    ap.add_argument('--emit-tok', type=int, default=256)
    ap.add_argument('--no-emit', action='store_true')
    a = ap.parse_args()

    a.ckpt = os.path.expanduser(a.ckpt)      # resolve ~ (Windows/PowerShell + Linux)
    model, enc, dev = load(a.ckpt, a.gpu)
    result = {'ckpt': a.ckpt}
    result['metric1_junkLL'] = metric1(model, enc, dev, a.junk, a.control, a.max_seq_tok)
    if not a.no_emit:
        result['metric2_emission'] = metric2(model, enc, dev, a.junk, a.emit, a.emit_tok)
    json.dump(result, open(a.out, 'w'), indent=1)
    m1 = result['metric1_junkLL']
    print(f"[{os.path.basename(a.ckpt.rstrip('/'))}] junk_nll/tok={m1['junk_mean_nll_per_tok']:.4f} "
          f"ctrl_nll/tok={m1['ctrl_mean_nll_per_tok']} "
          f"emission={result.get('metric2_emission',{}).get('emission_rate')}")

if __name__ == '__main__':
    main()
