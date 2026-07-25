#!/usr/bin/env python3
"""Unit test within_dedup_filter against the survey's known furniture vs content."""
import sys; sys.path.insert(0,'.')
from within_dedup_filter import within_dedup_book, is_protected, is_furniture, norm, _degenerate

def rep(line, n):  # realistic book: `line` repeated n times, diluted in 5000 distinct prose lines
    body = [f"This is distinct sentence number {i} carrying real unique narrative content here."
            for i in range(5000)]
    out = list(body)
    for i in range(n):        # sprinkle the furniture line through the prose (page furniture is sparse)
        out.insert(i * (len(out) // max(n, 1)), line)
    return '\n'.join(out)

FURNITURE = [   # (line, repeat_count) that SHOULD be cut
    ('---|---', 156),
    ('1234| **56.78**', 456),
    ('123.', 397),
    ('45.', 89),
    ('Corrie Wingate/APA Publications', 152),
    ('iStock', 36),
    ('fig. 12.3', 60),
    ('s.1(2)| 34–567', 60),
    ('**(1.2)**', 81),
]
CONTENT = [     # (line, repeat_count) that MUST survive
    ('* * *', 896),
    ('Preheat the oven to 350°F (175°C).', 34),
    ('"No."', 6),
    ('## Summary', 7),
    ('HAMLET:', 300),
    ('FIRST WITCH', 200),
    ('1ST MURDERER', 150),
    ('Cellar sales:', 400),
    ('ACT I', 60),
    ('_bibliography_', 1199),   # single italic word heading — protect (word, not furniture-shaped)
]

def cut_test(line, n):
    txt = rep(line, n)
    out, info = within_dedup_book(txt)
    removed = line in txt and line not in out.split('\n')
    return removed, info

print("=== FURNITURE (want CUT) ===")
fok = 0
for line, n in FURNITURE:
    removed, info = cut_test(line, n)
    ok = removed
    fok += ok
    print(f"  [{'CUT ' if removed else 'KEPT'}] {'ok' if ok else 'MISS':4} x{n:<4} {line!r}  {info.get('reasons',info)}")

print("\n=== CONTENT (want KEEP) ===")
cok = 0
for line, n in CONTENT:
    removed, info = cut_test(line, n)
    ok = not removed
    cok += ok
    print(f"  [{'CUT ' if removed else 'KEPT'}] {'ok' if ok else 'EATEN!':6} x{n:<4} {line!r}")

print("\n=== DEGENERATE within-line (want CUT) ===")
deg = ['here ' * 2811, 'ref 1 ref 2 ref 3 ref 4 ref 5 ' * 100, '. ' * 200]
for d in deg:
    print(f"  [{'CUT ' if _degenerate(d) else 'KEPT'}] tokens={len(d.split()):<6} {d[:40]!r}...")
prose = "The quick brown fox jumped over the lazy dog while the sun slowly set behind distant rolling hills."
print(f"  [{'CUT ' if _degenerate(prose*3) else 'KEPT'}] (long prose, want KEPT) {prose[:40]!r}...")

print(f"\nFURNITURE cut {fok}/{len(FURNITURE)}   CONTENT kept {cok}/{len(CONTENT)}")
