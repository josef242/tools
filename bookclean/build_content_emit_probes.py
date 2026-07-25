#!/usr/bin/env python3
"""Content-prose emission probes (Rook's free corollary): held-out CLEAN prose spans long
enough to split prefix(24)+continuation(48), for greedy-verbatim-overlap at each model size.
Held-out = book.v11.shuf.jsonl docs after the 18k training prefix (neither arm trained on).
Measures CONTENT memorization scaling with capacity. Output: reports/content_emit_probes.jsonl"""
import json, os, re, random
SRC = os.path.expanduser('~/data/book.v11.shuf.jsonl'); TRAIN_DOCS = 18000
OUT = 'reports/content_emit_probes.jsonl'; N = 400
rng = random.Random(7)
def is_prose_line(s): return len(re.findall(r'[a-z]{2,}', s.strip().lower())) >= 5
out = []
with open(SRC) as f:
    for lineno, line in enumerate(f):
        if lineno < TRAIN_DOCS: continue
        if len(out) >= N: break
        if rng.random() > 0.05: continue
        try: t = json.loads(line)['text']
        except Exception: continue
        # find a contiguous prose block >= 400 chars (enough for 72 tokens)
        for para in t.split('\n\n'):
            p = para.strip().replace('\n', ' ')
            if 400 <= len(p) <= 1200 and is_prose_line(p):
                out.append({'text': p[:1200], 'class': 'content'}); break
with open(OUT, 'w') as g:
    for r in out: g.write(json.dumps(r, ensure_ascii=True) + '\n')
print(f"content-emit probes: {len(out)} held-out prose spans -> {OUT}")
print("sample:", out[0]['text'][:110] if out else "(none)")
