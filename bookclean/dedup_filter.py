#!/usr/bin/env python3
"""Dedup filter v1 (SAFE tier): remove cross-book boilerplate LINES corpus-wide.
A line is removed iff (STRONG boilerplate pattern) AND (appears in >=3 distinct
books, digit-normalized). Per-book CIRCUIT BREAKER: if removals exceed 25% of a
book's non-empty lines, remove NOTHING and flag the book (protects
cookbooks/anthologies/quotation-dictionaries; flagged list = failure-genre mining).
Tier 2 (length+multiplicity without a named pattern) deferred — the suspect audit
showed it carries content risk (recipes/dialogue) the safe tier avoids."""
import re, hashlib
def norm(l):
    l=re.sub(r'\s+',' ',l.strip().lower()); return re.sub(r'\d','#',l)
def hh(l): return hashlib.blake2b(norm(l).encode(),digest_size=8).digest()

# STRONG boilerplate line patterns (expanded via the >=100-book suspect audit).
STRONG_LINE = re.compile(
    r"\bISBN\b|e-?ISBN|©|\bcopyright\b|all rights reserved|"
    r"library of congress|cataloguing?[- ]in[- ]publication|catalogue record|"
    r"(?:first |electronic |this )?(?:e-?)?edition (?:published|first)|e-?library|"
    r"(?:manufactured|printed|typeset)(?: and bound)? in\b|"
    r"a division of|an imprint of|published (?:by|simultaneously)|"
    r"\bpenguin\b|harpercollins|macmillan|hachette|simon & schuster|random house|"
    r"bloomsbury|houghton mifflin|scholastic|routledge|taylor & francis|"
    r"we hope you (?:have )?enjoyed|thank you for (?:purchasing|buying|reading)|"
    r"sign up (?:for|to)|our newsletter|exclusive (?:gifts|content)|meet the authors|"
    r"watch videos|personalized book picks|up-to-date news about this author|"
    r"for the best (?:in paperbacks|reading experience)|"
    r"typographical errors have been corrected|right of the author to be identified|"
    r"moral rights? of the author|acquisitions, editorial|"
    r"reproduced.{0,30}(?:retrieval system|any form|written permission)|"
    r"project gutenberg|michael hart|"
    r"visit (?:us|our)|our website|connect with (?:us|the author)",
    re.I)

BREAKER_FRAC = 0.25          # >25% of a book's lines -> flag, don't filter
MIN_DUP = 3                  # line must be boilerplate in >=3 books

def dedup_book(text, freq, min_dup=MIN_DUP):
    lines = text.split('\n')
    nonempty = [i for i,l in enumerate(lines) if l.strip()]
    remove = []
    for i in nonempty:
        s = lines[i].strip()
        if 12 <= len(s) <= 250 and STRONG_LINE.search(s) and freq.get(hh(s),0) >= min_dup:
            remove.append(i)
    if not remove:
        return text, {'removed': 0}
    if len(remove) > BREAKER_FRAC * max(len(nonempty),1):
        return text, {'flagged': True, 'reason': f'{len(remove)}/{len(nonempty)} lines',
                      'removed': 0}
    rm = set(remove)
    kept = [l for i,l in enumerate(lines) if i not in rm]
    out = re.sub(r'\n{4,}', '\n\n\n', '\n'.join(kept))
    return out, {'removed': len(remove),
                 'sample': lines[remove[0]].strip()[:80]}
