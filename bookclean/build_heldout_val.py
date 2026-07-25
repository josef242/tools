#!/usr/bin/env python3
"""Held-out clean-val set (opportunity-cost / displacement probe, Rook pre-reg NULL).

Both arms trained on the SAME 18,000-doc prefix of book.v11.shuf.jsonl (arm A = v9 versions,
arm B = v11 versions of those same books). So any doc from line 18001+ of the shuffle was seen
by NEITHER arm -> genuine held-out CLEAN prose. We sample prose spans from that region and score
both arms' final (3-epoch) checkpoints; the A-B NLL gap on this set measures whether arm B (which
spent its fixed token budget entirely on clean prose) generalizes better to unseen clean prose than
arm A (which spent ~0.1% of budget on junk). Pre-registered prediction: NULL (|A-B| within seed noise).

Length-matched to the content-control distribution for comparability. Output: reports/heldout_val.jsonl
(schema {text, lines}, same as content_control.jsonl)."""
import json, os, re, random

SRC = os.path.expanduser('~/data/book.v11.shuf.jsonl')
TRAIN_DOCS = 18000                 # both arms trained on the 18k prefix; held-out = docs after it
OUT = 'reports/heldout_val.jsonl'
N_TARGET = 6000
rng = random.Random(4242)          # same seed family as content_control

# target char-length distribution from the junk reservoir (matches content_control)
junk_lens = [len(json.loads(l)['text']) for l in open('reports/junk_lexicon.jsonl')]
junk_lens.sort()

def is_prose_line(s):
    s = s.strip()
    return len(re.findall(r'[a-z]{2,}', s.lower())) >= 4   # real words -> content, not furniture

def sample_span(rec_lines, target_len):
    ne = [i for i, l in enumerate(rec_lines) if l.strip()]
    if len(ne) < 2:
        return None
    start = rng.choice(ne)
    span, clen = [], 0
    for i in range(start, len(rec_lines)):
        l = rec_lines[i]
        span.append(l); clen += len(l) + 1
        if clen >= target_len:
            break
    txt = '\n'.join(span).strip()
    # require the span to be mostly prose (>=60% prose lines), else resample-skip
    nonempty = [l for l in span if l.strip()]
    if not nonempty:
        return None
    if sum(1 for l in nonempty if is_prose_line(l)) / len(nonempty) < 0.6:
        return None
    return txt

def main():
    out = []
    seen_docs = 0
    with open(SRC) as f:
        for lineno, line in enumerate(f):
            if lineno < TRAIN_DOCS:          # skip the training prefix
                continue
            if len(out) >= N_TARGET:
                break
            if rng.random() > 0.05:          # sample ~5% of held-out docs
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            t = rec.get('text')
            if not isinstance(t, str):
                continue
            seen_docs += 1
            tlen = junk_lens[rng.randrange(len(junk_lens))]
            s = sample_span(t.split('\n'), tlen)
            if s and len(s) >= 40:
                out.append({'text': s, 'lines': s.count('\n') + 1})
    with open(OUT, 'w') as g:
        for r in out:
            g.write(json.dumps(r, ensure_ascii=True) + '\n')
    print(f"held-out clean-val: {len(out)} spans from {seen_docs} held-out docs (source lines {TRAIN_DOCS+1}+) -> {OUT}")
    print("sample:", out[0]['text'][:120].replace('\n', ' / ') if out else "(none)")

if __name__ == '__main__':
    main()
