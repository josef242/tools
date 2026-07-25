#!/usr/bin/env python3
"""Matched content control for the junk-LL metric. Samples CONTENT spans from v11
(kept text -> present in BOTH ablation arms), length-matched to the junk lexicon's
span-length distribution. Because these spans are in both arms, arms A and B should
assign near-identical NLL to them -> isolates that any junk-NLL gap is the removal,
not a generic length/style effect. Output: reports/content_control.jsonl (same
schema as junk_lexicon.jsonl: {text, lines})."""
import json, os, random, re
V11 = os.path.expanduser('~/data/book.v11.shuf.jsonl')
N = 8000
rng = random.Random(4242)

# target char-length distribution from the junk reservoir
junk_lens = [len(json.loads(l)['text']) for l in open('reports/junk_lexicon.jsonl')]
junk_lens.sort()

def is_prose_line(s):
    s = s.strip()
    return len(re.findall(r'[a-z]{2,}', s.lower())) >= 4  # real words -> content, not furniture

def sample_span(rec_lines, target_len):
    ne = [i for i, l in enumerate(rec_lines) if l.strip()]
    if len(ne) < 2:
        return None
    start = rng.choice(ne)
    span = []; clen = 0
    for i in range(start, len(rec_lines)):
        l = rec_lines[i]
        span.append(l); clen += len(l) + 1
        if clen >= target_len:
            break
    text = '\n'.join(span).strip()
    # require the span to be mostly prose (control must be real content, not residual junk)
    pl = [l for l in span if l.strip()]
    if not pl or sum(is_prose_line(l) for l in pl) / len(pl) < 0.6:
        return None
    return text, len([l for l in span if l.strip()])

def main():
    out = open('reports/content_control.jsonl', 'w')
    kept = 0
    # stream v11 (already shuffled -> spread across corpus), parse a sampled subset
    with open(V11) as f:
        for idx, line in enumerate(f):
            if kept >= N:
                break
            if rng.random() > 0.02:           # sample ~2% of records
                continue
            try:
                text = json.loads(line)['text']
            except Exception:
                continue
            lines = text.split('\n')
            tlen = junk_lens[rng.randrange(len(junk_lens))]
            s = sample_span(lines, tlen)
            if s:
                out.write(json.dumps({'text': s[0], 'lines': s[1]}, ensure_ascii=True) + '\n')
                kept += 1
    out.close()
    print(f"content control: {kept} spans -> reports/content_control.jsonl")

if __name__ == '__main__':
    main()
