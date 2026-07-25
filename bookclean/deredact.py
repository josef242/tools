#!/usr/bin/env python3
"""De-redaction of PG19 <DWnn> tokens in the RedPajama book corpus.

Restores the original words destroyed by DeepMind's Ofcom-list profanity
filter, using the ground-truthed mapping in reports/dw_mapping.json.

Restoration is deterministic at the word level (token -> word). Capitalization
was not preserved by the filter, so it is restored by tiered heuristics:

  1. special contexts   (island of Negros, Cur Deus Homo, Coon Creek, Div.,
                         Honky Tonk, Lesbos, O'-prefixed surnames)
  2. ALL-CAPS window    -> UPPERCASE
  3. sentence/line start-> Capitalized
  4. title prefix       (Mr./Dr./Captain/... immediately before) -> Capitalized
  5. per-word default   (ethnonyms usually capitalized in period texts: Jap,
                         Kafir, Polack, Paki, ... ; common nouns lowercase)

A trailing 's' on the token means the filter consumed a plural: it is restored
with the correct English plural (darky->darkies, negro->negroes, ...).

DW52 maps to both spellings (coloured/colored); the book's own orthography
(colour/honour/labour vs color/honor/labor counts) decides which to restore.

Every replacement is written to a ledger (JSONL) with context and the rule
that fired, so the pass is fully auditable and reversible.
"""

import json
import re
import os

MAPPING = {
    2: "fop", 5: "kafir", 6: "kike", 7: "papist", 8: "prod", 9: "taig",
    10: "yid", 12: "bender", 15: "bummer", 18: "dyke", 19: "faggot",
    25: "homo", 26: "lesbo", 29: "pansy", 30: "poof", 35: "cretin",
    36: "cripple", 37: "div", 38: "loony", 40: "midget", 41: "mong",
    42: "nutter", 43: "psycho", 44: "retard", 45: "schizo", 46: "spastic",
    50: "chinky", 52: "coloured", 53: "coon", 54: "darky", 55: "dago",
    56: "gippo", 57: "golliwog", 58: "gook", 60: "honky", 61: "jap",
    62: "kraut", 64: "negro", 65: "nigger", 67: "paki", 68: "pikey",
    69: "polack", 71: "sambo", 72: "slope", 74: "spic", 75: "taff",
    76: "wog", 77: "wop",
}

PLURALS = {
    "chinky": "chinkies", "darky": "darkies", "pansy": "pansies",
    "loony": "loonies", "negro": "negroes", "dago": "dagoes",
    "lesbo": "lesbos",  # bare fallback; Lesbos island handled as special
}

# per-word mid-sentence case defaults, measured empirically against 36
# ground-truth book pairs. "negro" is handled adaptively per book: race-topic
# nonfiction (many DW64 tokens, e.g. Baker 1908 with 2000+) capitalizes it,
# fiction/memoir (few tokens) lowercases it - book-weighted truth is 13/16
# lowercase but token-weighted is dominated by the capitalizing books.
CAP_DEFAULT = {"kafir", "yid", "jap", "paki", "polack", "sambo",
               "wog", "dago", "papist", "lesbo"}
NEGRO_CAP_MIN_TOKENS = 40  # books with >= this many DW64 capitalize Negro

TITLE_RE = re.compile(
    r"(?:Mr|Mrs|Dr|Miss|Master|Captain|Capt|General|Gen|Colonel|Col|Major|"
    r"Lord|Lady|Sir|Father|Uncle|Aunt|Judge|Professor|Prof|Chief)\.?\s+$")

TOKEN_RE = re.compile(r"<DW(\d+)>(s?)")

BRIT_RE = re.compile(r"\b(?:colour|honour|labour|favour|neighbour)", re.I)
US_RE = re.compile(r"\b(?:color|honor|labor|favor|neighbor)", re.I)

SENT_SKIP = set("\"'“‘([*_ \t")
SENT_END = set(".!?\n")


def plural_of(word):
    return PLURALS.get(word, word + "s")


def is_allcaps_window(text, start, end, radius=28):
    window = text[max(0, start - radius):end + radius]
    letters = [c for c in window if c.isalpha()]
    return len(letters) >= 4 and not any(c.islower() for c in letters)


def at_sentence_start(text, pos):
    """True if pos begins a sentence. A single newline is a hard wrap (PG19
    wraps prose at ~70 cols) and is treated as a space; a blank line is a
    paragraph break."""
    i = pos - 1
    while i >= 0 and text[i] in SENT_SKIP:
        i -= 1
    if i >= 0 and text[i] == "\n":
        j = i - 1
        while j >= 0 and text[j] in " \t":
            j -= 1
        if j < 0 or text[j] == "\n":
            return True  # paragraph break / start of text
        i = j
        while i >= 0 and text[i] in SENT_SKIP:
            i -= 1
    if i < 0:
        return True
    return text[i] in ".!?"


def in_titlecase_line(text, pos):
    """True if the surrounding line looks like a heading/TOC entry
    (>=3 words, >=60% of them capitalized)."""
    ls = text.rfind("\n", 0, pos) + 1
    le = text.find("\n", pos)
    if le == -1:
        le = len(text)
    words = re.findall(r"[A-Za-z][\w']*", text[ls:pos] + text[pos:le])
    if len(words) < 3:
        return False
    caps = sum(1 for w in words if w[0].isupper())
    return caps / len(words) >= 0.6


def in_hyphen_compound(text, s, e):
    """taff-rail, sauer-kraut, psycho-analysis: lowercase compound member."""
    if text[e:e + 1] == "-" and text[e + 1:e + 2].islower():
        return True
    return s >= 2 and text[s - 1] == "-" and text[s - 2].islower()


def restore_text(text, book_spelling=None, ledger=None, book_idx=None):
    """Replace all <DWnn> tokens in text; returns (new_text, n_replaced)."""
    n = 0
    cap_words = CAP_DEFAULT | ({"negro"} if text.count("<DW64>") >= NEGRO_CAP_MIN_TOKENS else set())

    def repl(m):
        nonlocal n
        tid = int(m.group(1))
        plural = bool(m.group(2))
        word = MAPPING.get(tid)
        if word is None:  # unknown id: leave untouched
            return m.group(0)
        if tid == 52 and book_spelling == "us":
            word = "colored"
        s, e = m.start(), m.end()
        rule = "default"
        out = None

        # --- special contexts ---
        if tid == 26 and plural:
            out, rule = "Lesbos", "special:lesbos"
        elif tid == 64 and plural and text[max(0, s - 12):s].lower().endswith("island of "):
            out, rule = "Negros", "special:negros-island"
        elif tid == 25 and (text[max(0, s - 5):s] == "Deus " or
                            re.match(r"\s+(?:Anthropos|Pithekos|Sapiens)\b", text[e:e + 12])):
            out, rule = "Homo", "special:latin-homo"
        elif tid == 53 and re.match(r"\s+Creek\b", text[e:e + 8]):
            out, rule = "Coon", "special:coon-creek"
        elif tid == 37 and text[e:e + 1] == "." and re.search(r"[A-Z][\w.&]*\s+$", text[max(0, s - 16):s]):
            out, rule = "Div", "special:div-abbrev"
        elif tid == 60 and re.match(r"[\s-]+Tonk\b", text[e:e + 8]):
            out, rule = "Honky", "special:honky-tonk"
        elif text[max(0, s - 2):s] in ("O'", "O’"):
            out, rule = word.capitalize(), "oprefix"

        if out is None:
            base = plural_of(word) if plural else word
            if is_allcaps_window(text, s, e):
                out, rule = base.upper(), "allcaps"
            elif in_hyphen_compound(text, s, e):
                out, rule = base, "hyphen-compound"
            elif at_sentence_start(text, s):
                out, rule = base.capitalize(), "sentence-start"
            elif in_titlecase_line(text, s):
                out, rule = base.capitalize(), "titlecase-line"
            elif TITLE_RE.search(text[max(0, s - 14):s]):
                out, rule = base.capitalize(), "title-prefix"
            elif word in cap_words:
                out, rule = base.capitalize(), "cap-default"
            else:
                out = base
        elif plural and rule not in ("special:lesbos", "special:negros-island"):
            out = plural_of(out.lower()).capitalize() if out[0].isupper() else plural_of(out)

        n += 1
        if ledger is not None:
            ledger.write(json.dumps({
                "book": book_idx, "id": tid, "restored": out, "rule": rule,
                "ctx": text[max(0, s - 40):s] + "[" + out + "]" + text[e:e + 40],
            }) + "\n")
        return out

    return TOKEN_RE.sub(repl, text), n


def book_spelling_of(text):
    brit = len(BRIT_RE.findall(text))
    us = len(US_RE.findall(text))
    return "us" if us > brit else "brit"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/data/book.jsonl"))
    ap.add_argument("--dst", default=os.path.expanduser("~/data/book.v1.jsonl"))
    ap.add_argument("--ledger", default="reports/deredact_ledger.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.ledger) or ".", exist_ok=True)
    total = books_touched = 0
    with open(args.src, "rb") as src, open(args.dst, "wb") as dst, \
         open(args.ledger, "w") as ledger:
        for idx, raw in enumerate(src):
            if b"<DW" not in raw:
                dst.write(raw)          # fast path: untouched books byte-identical
                continue
            rec = json.loads(raw)
            text = rec.get("text", "")
            spelling = book_spelling_of(text) if "<DW52>" in text else None
            new_text, n = restore_text(text, spelling, ledger, idx)
            if n:
                rec["text"] = new_text
                books_touched += 1
                total += n
                dst.write(json.dumps(rec, ensure_ascii=True).encode() + b"\n")
            else:
                dst.write(raw)
            if books_touched and books_touched % 2000 == 0:
                print(f"  {books_touched:,} books de-redacted, {total:,} tokens", flush=True)
    print(f"DONE: {total:,} tokens restored across {books_touched:,} books -> {args.dst}")


if __name__ == "__main__":
    main()
