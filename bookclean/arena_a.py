#!/usr/bin/env python3
"""Arena A: ocr/stage2b engine on bookclean turf.

Adapter: chunk a head window into pseudo-pages (~PAGE_CHARS, split at paragraph
boundaries), run stage2b's extract_features + classify_page, then apply its
boundary contract (body_start = page after the LATEST HIGH-front cluster) and
map back to a char offset.
"""
import json, sys
sys.path.insert(0, '/home/josef/valhalla/code/ocr')
sys.path.insert(0, '.')
import stage2b_detect_boilerplate as s2b
from bakeoff import paragraphs
from sentinel2 import head_boundary as sentinel2_head, P as TUNED
from sentinel3 import asymmetric_cost

PAGE_CHARS = 2000

def to_pages(text):
    """[(start_offset, page_text)] — paragraph-aligned pseudo-pages."""
    pages, cur, cur_start, cur_len = [], [], 0, 0
    for s, e, txt in paragraphs(text):
        if cur and cur_len + (e - s) > PAGE_CHARS:
            pages.append((cur_start, "\n\n".join(cur)))
            cur, cur_start, cur_len = [], s, 0
        if not cur:
            cur_start = s
        cur.append(txt)
        cur_len += e - s
    if cur:
        pages.append((cur_start, "\n\n".join(cur)))
    return pages

def stage2b_head(text, title="", author=""):
    """content_start char offset per stage2b's front contract."""
    pages = to_pages(text)
    if not pages:
        return 0
    last_high_front = -1
    for i, (off, ptext) in enumerate(pages):
        f = s2b.extract_features(ptext, title, author, i + 1)
        cls = s2b.classify_page(f, position_hint="front_sample")
        if cls.kind == "front" and cls.confidence == "high":
            last_high_front = i
    if last_high_front < 0:
        return 0                       # no HIGH-front page: book starts clean
    if last_high_front + 1 >= len(pages):
        return len(text)               # everything is front matter
    return pages[last_high_front + 1][0]

if __name__ == "__main__":
    w2 = {b['offset']: b for b in json.load(open('reports/gold_wave2_books.json'))}
    w3 = {b['offset']: b for b in json.load(open('reports/gold_wave3_books.json'))}
    pilot = {b['offset']: b for b in json.load(open('reports/gold_pilot_books.json'))}
    gold = json.load(open('reports/gold_labels.json'))

    rows = []
    for g in gold:
        src = w2.get(g['offset']) or w3.get(g['offset']) or pilot.get(g['offset'])
        head = src['head']
        truth = len(head) if g['content_start'] == -1 else g['content_start']
        s2 = sentinel2_head(head, TUNED)
        s2 = s2 if s2 is not None else len(head)
        ocr = stage2b_head(head, src.get('title', ''))
        rows.append((truth, s2, ocr))

    def report(name, preds):
        errs = [p - t for t, p in preds]
        n = len(errs)
        med = sorted(abs(e) for e in errs)[n // 2]
        cost = sum(asymmetric_cost(p, t, 'head') for t, p in preds) / n
        cut_books = sum(1 for e in errs if e > 200)      # content eaten
        left_books = sum(1 for e in errs if e < -500)    # junk left
        tight = sum(1 for e in errs if abs(e) <= 150)
        cut_chars = sum(max(0, e) for e in errs)
        left_chars = sum(max(0, -e) for e in errs)
        print(f"{name:<12} medErr={med:<6} tight={tight:>2}/{n}  "
              f"CUT_books={cut_books:<2} LEFT_books={left_books:<2}  "
              f"cut_chars={cut_chars:>7,} left_chars={left_chars:>7,}  "
              f"asym_cost={cost:.2f}")

    print(f"\n=== ARENA A: heads, {len(rows)} gold labels (policy-compatible) ===")
    print("asym_cost = (2.0*content_cut + 0.1*junk_left) / 100 chars, per book\n")
    report("sentinel2", [(t, s2) for t, s2, _ in rows])
    report("ocr/stage2b", [(t, o) for t, _, o in rows])
