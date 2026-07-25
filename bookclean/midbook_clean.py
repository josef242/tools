#!/usr/bin/env python3
"""v4 mid-book passes: structured transcriber notes, [Illustration]/[Sidenote]
tags (including captions - they duplicate adjacent text). Corpus-wide."""
import re

TN_BRACKET = re.compile(r'\[\s*transcriber\W{0,3}s?\s*notes?\b[^\]]{0,2500}\]', re.I | re.S)
TN_PARA = re.compile(r'(?:^|\n\n)\s*transcriber\W{0,3}s?\s*notes?\s*[:.](?:(?!\n\n).){0,2500}', re.I | re.S)
ILLUS = re.compile(r'\[\s*illustration[^\]]{0,1200}\]', re.I | re.S)
SIDENOTE = re.compile(r'\[\s*sidenote[^\]]{0,600}\]', re.I | re.S)
SQUEEZE = re.compile(r'\n{4,}')

def clean_midbook(text):
    counts = {}
    for name, rx in (('tn_bracket', TN_BRACKET), ('tn_para', TN_PARA),
                     ('illustration', ILLUS), ('sidenote', SIDENOTE)):
        text, n = rx.subn('', text)
        if n: counts[name] = n
    if counts:
        text = SQUEEZE.sub('\n\n\n', text)
    return text, counts

if __name__ == '__main__':
    import json, os, random
    rng = random.Random(11)
    rows = []
    for line in open('reports/book_index_v1.tsv'):
        i, off, ln, reg, title = line.rstrip('\n').split('\t')
        rows.append((int(i), reg, title))
    # need offsets into v3: stream-sample instead
    from collections import Counter
    totals = Counter(); books_touched = 0; n = 0
    examples = []
    with open(os.path.expanduser('~/data/book.v3.jsonl'), 'rb') as f:
        for raw in f:
            if rng.random() > 0.004: continue
            n += 1
            rec = json.loads(raw)
            t2, counts = clean_midbook(rec['text'])
            if counts:
                books_touched += 1
                totals.update(counts)
                if len(examples) < 5 and 'tn_bracket' in counts:
                    m = TN_BRACKET.search(rec['text'])
                    examples.append(' '.join(m.group().split())[:130])
            if n >= 800: break
    print(f"sampled {n} books, touched {books_touched} ({100*books_touched/n:.0f}%)")
    print(dict(totals))
    print("\nremoved transcriber-note examples:")
    for x in examples: print("  |", x)
