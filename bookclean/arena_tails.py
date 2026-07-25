#!/usr/bin/env python3
"""Tail bake-off under the structural-hybrid policy.

Contenders:
  sentinel2   — v8 production tail engine (cuts ALL back matter)
  sentinel3   — hybrid single-cut (cuts data-structures/marketing only)
  sentinel3-span — hybrid MULTI-SPAN: excises boilerplate blocks, keeps prose
                   islands between them (handles non-monotone back matter)
  stage2b     — ocr engine, pseudo-page adapter, its own back contract
Metric: asymmetric cost (2.0 * content cut + 0.1 * junk left) per 100 chars.
"""
import json, sys
sys.path.insert(0, '/home/josef/valhalla/code/ocr')
sys.path.insert(0, '.')
import stage2b_detect_boilerplate as s2b
from bakeoff import paragraphs, sentinel_tail as sentinel2_tail
from sentinel2 import tail_boundary as sentinel2_tail_v22
from sentinel3 import tail_boundary as sentinel3_tail, classify_tail_para, asymmetric_cost
from arena_a import to_pages


def sentinel3_span(tail):
    """Multi-span: returns list of (start,end) CUT spans. Prose islands survive.
    Only merges adjacent cut/neutral runs that contain a HIGH-confidence signal."""
    paras = paragraphs(tail)
    if not paras:
        return []
    kinds = [classify_tail_para(t) for _, _, t in paras]
    spans, i = [], 0
    while i < len(paras):
        if kinds[i] == 'cut':
            j = i
            # extend over cut/neutral, stop at first keep
            while j + 1 < len(paras) and kinds[j + 1] in ('cut', 'neutral'):
                j += 1
            # trim trailing neutrals back to the last real cut
            k = j
            while k > i and kinds[k] != 'cut':
                k -= 1
            spans.append((paras[i][0], paras[k][1]))
            i = j + 1
        else:
            i += 1
    return spans


def stage2b_tail(tail, title=""):
    """content_end via stage2b's back contract: page before EARLIEST HIGH-back."""
    pages = to_pages(tail)
    if not pages:
        return len(tail)
    first_high_back = -1
    for i, (off, ptext) in enumerate(pages):
        f = s2b.extract_features(ptext, title, "", i + 1)
        cls = s2b.classify_page(f, position_hint="back_sample")
        if cls.kind == "back" and cls.confidence == "high":
            first_high_back = i
            break
    if first_high_back < 0:
        return len(tail)
    return pages[first_high_back][0]


def span_cost(spans, truth, tail_len):
    """Asymmetric cost for a multi-span cutter. Content = [0,truth); junk = [truth,len)."""
    cut_content = sum(max(0, min(e, truth) - min(s, truth)) for s, e in spans)
    cut_junk = sum(max(0, e - max(s, truth)) for s, e in spans if e > truth)
    junk_total = tail_len - truth
    left_junk = junk_total - cut_junk
    return (2.0 * cut_content + 0.1 * max(0, left_junk)) / 100, cut_content, left_junk


if __name__ == "__main__":
    books = {b['offset']: b for b in json.load(open('reports/gold_tails_books.json'))}
    hyb = json.load(open('reports/gold_tail_labels_hybrid.json'))
    old = {g['offset']: g['content_end'] for g in json.load(open('reports/gold_tail_labels.json'))}

    results = {k: [] for k in ('sentinel2', 'sentinel3', 'sentinel3-span', 'stage2b')}
    policy_shift = []
    for g in hyb:
        tail = books[g['offset']]['tail']
        truth = g['content_end']
        L = len(tail)
        policy_shift.append(truth - old.get(g['offset'], truth))
        for name, pred in (('sentinel2', sentinel2_tail_v22(tail)),
                           ('sentinel3', sentinel3_tail(tail)),
                           ('stage2b', stage2b_tail(tail, books[g['offset']].get('title', '')))):
            c = asymmetric_cost(pred, truth, 'tail')
            cut = max(0, truth - pred)
            left = max(0, pred - truth)
            results[name].append((c, cut, left))
        spans = sentinel3_span(tail)
        c, cut, left = span_cost(spans, truth, L)
        results['sentinel3-span'].append((c, cut, left))

    n = len(hyb)
    print(f"\n=== TAIL BAKE-OFF — structural-hybrid policy, {n} gold labels ===")
    print("asym_cost = (2.0*content_cut + 0.1*junk_left)/100 chars, mean per book\n")
    print(f"{'engine':<16}{'asym_cost':>10}{'content_cut':>13}{'junk_left':>11}{'books_cut':>11}{'clean':>7}")
    for name, rows in results.items():
        cost = sum(r[0] for r in rows) / n
        cut = sum(r[1] for r in rows)
        left = sum(r[2] for r in rows)
        books_cut = sum(1 for r in rows if r[1] > 200)
        clean = sum(1 for r in rows if r[1] == 0)
        print(f"{name:<16}{cost:>10.2f}{cut:>13,}{left:>11,}{books_cut:>11}{clean:>7}/{n}")
    print(f"\npolicy shift (hybrid content_end - old content_end): "
          f"mean +{sum(policy_shift)/len(policy_shift):.0f} chars kept that v8 cut")
