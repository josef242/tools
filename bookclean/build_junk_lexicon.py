#!/usr/bin/env python3
"""Build the ablation's junk lexicon = the v9->v11 diff (the exact text arms A and
B differ by). v11 is v9 with junk LINES deleted (dedup never inserts), so v11's
non-empty lines are a subsequence of v9's -> a two-pointer walk recovers the removed
lines AND their contiguity (contiguous removed runs = junk SPANS, the right unit for
teacher-forced log-likelihood eval). Metric #1 of ABLATION_PREREG.md.

Outputs (reports/):
  junk_lexicon.jsonl   reservoir sample of removed junk SPANS (list of lines each)
  junk_lexicon_stats.json  totals: removed chars/lines, span-length histogram
This is the free labeled junk set the reversible ledger gives us -- ~120M chars of
positives, zero annotation."""
import json, os, sys, random
V9  = os.path.expanduser('~/data/book.v9.jsonl')
V11 = os.path.expanduser('~/data/book.v11.jsonl')
CAP = 200_000                      # reservoir size (spans)
MIN_SPAN_CHARS = 24                # ignore trivial 1-token removals as eval sequences
rng = random.Random(20260716)

def removed_spans(t9, t11):
    """Contiguous runs of v9 lines absent from v11 (subsequence two-pointer)."""
    a = [l for l in t9.split('\n') if l.strip()]
    b = [l for l in t11.split('\n') if l.strip()]
    i = j = 0; cur = []; spans = []
    while i < len(a):
        if j < len(b) and a[i] == b[j]:
            if cur:
                spans.append(cur); cur = []
            i += 1; j += 1
        else:
            cur.append(a[i]); i += 1
    if cur:
        spans.append(cur)
    return spans

def main():
    reservoir = []; seen = 0
    tot_chars = tot_lines = tot_spans = nb = 0
    hist = {}                                    # span line-count -> count
    with open(V9) as f9, open(V11) as f11, \
         open('reports/junk_lexicon.jsonl', 'w') as out:
        for l9, l11 in zip(f9, f11):
            nb += 1
            r9 = json.loads(l9); r11 = json.loads(l11)
            if len(r9['text']) == len(r11['text']):
                continue                          # untouched record, cheap skip
            for span in removed_spans(r9['text'], r11['text']):
                text = '\n'.join(span)
                tot_spans += 1; tot_lines += len(span); tot_chars += len(text)
                hist[len(span)] = hist.get(len(span), 0) + 1
                if len(text) < MIN_SPAN_CHARS:
                    continue
                seen += 1
                item = {'text': text, 'lines': len(span)}
                if len(reservoir) < CAP:
                    reservoir.append(item)
                else:                             # reservoir sampling
                    k = rng.randint(0, seen - 1)
                    if k < CAP:
                        reservoir[k] = item
            if nb % 20000 == 0:
                print(f"  {nb} records | {tot_spans:,} spans | {tot_chars/1e6:.1f}M junk chars",
                      flush=True)
        for it in reservoir:
            out.write(json.dumps(it, ensure_ascii=True) + '\n')
    json.dump({'records': nb, 'removed_spans': tot_spans, 'removed_lines': tot_lines,
               'removed_chars': tot_chars, 'reservoir': len(reservoir),
               'span_len_hist': dict(sorted(hist.items())[:40])},
              open('reports/junk_lexicon_stats.json', 'w'), indent=1)
    print(f"DONE: {tot_spans:,} spans, {tot_chars/1e6:.1f}M junk chars, "
          f"reservoir={len(reservoir):,} -> reports/junk_lexicon.jsonl")

if __name__ == '__main__':
    main()
