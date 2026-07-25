#!/usr/bin/env python3
"""Build the emission-probe set: junk spans long enough to split into a prefix
(context) + a continuation (the part we test for verbatim emission). Tagged by the
source-defined class (junk_class.json) so the probe measures per-class: HARM class
(cross-book boilerplate) should emit under arm A; NULL class (furniture) should not.
Output: reports/emit_probes.jsonl  [{text, class}]."""
import json, random
lex = [json.loads(l)['text'] for l in open('reports/junk_lexicon.jsonl')]
cls = json.load(open('reports/junk_class.json'))
n = min(len(lex), len(cls))
rng = random.Random(11)
# want spans with enough tokens for prefix(24)+cont(48); ~90+ chars is a safe proxy
def ok(t): return 90 <= len(t) <= 600
cross = [i for i in range(n) if cls[i].startswith('CROSS') and ok(lex[i])]
within = [i for i in range(n) if cls[i].startswith('WITHIN') and ok(lex[i])]
rng.shuffle(cross); rng.shuffle(within)
probes = ([{'text': lex[i], 'class': 'cross-book'} for i in cross[:400]] +
          [{'text': lex[i], 'class': 'within-book'} for i in within[:200]])
with open('reports/emit_probes.jsonl', 'w') as f:
    for p in probes:
        f.write(json.dumps(p, ensure_ascii=True) + '\n')
print(f"emit probes: {len(probes)} ({min(len(cross),400)} cross-book + {min(len(within),200)} within-book)")
print('cross-book samples:', [lex[i][:50] for i in cross[:3]])
