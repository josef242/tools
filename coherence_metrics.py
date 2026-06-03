# coherence_metrics.py
#
# Pure-Python metrics for detecting text-degeneration signatures in generated
# output from an LLM checkpoint. No model dependencies — operates on generated
# text (plus optional per-token entropy / token-id arrays for the intrinsic
# metrics).
#
# Metric list (v1):
#   - seq_rep_n        : fraction of duplicate n-grams (Holtzman 2020)
#   - distinct_n       : unique-n-gram ratio (Li 2016)
#   - compression_ratio: gzip(len_compressed / len_raw)
#   - mattr            : moving-average type-token ratio, length-insensitive
#   - entity_persist   : total_mentions / unique_entities on capitalized words
#   - cross_span_entity: fraction of entities appearing in both first and last
#                        third of the generation
#   - entropy_ratio    : mean entropy on content tokens / mean entropy on
#                        function tokens (from the model's own per-token
#                        entropies during generation)

from __future__ import annotations

import gzip
import math
import re
from typing import Iterable, List, Optional, Sequence


# Top ~200 English function words. Anything in this set is treated as a
# "function" token for the entropy-ratio metric; everything else that passes
# the alpha-length gate is treated as a "content" token.
STOPWORDS = frozenset({
    # articles
    "the", "a", "an",
    # pronouns
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "mine", "yours", "hers", "ours", "theirs",
    "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "yourselves", "themselves",
    "this", "that", "these", "those",
    "who", "whom", "whose", "which", "what",
    # auxiliary + modal verbs
    "is", "am", "are", "was", "were", "be", "being", "been",
    "have", "has", "had", "having",
    "do", "does", "did", "doing", "done",
    "will", "would", "shall", "should", "may", "might", "must",
    "can", "could", "ought", "need", "dare", "used",
    # conjunctions
    "and", "or", "but", "if", "because", "as", "while", "when",
    "where", "whereas", "although", "though", "unless", "until",
    "since", "so", "yet", "for", "nor", "either", "neither", "both",
    "whether", "than", "whereupon", "wherever", "whenever",
    # prepositions
    "in", "on", "at", "by", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "out", "off", "over", "under",
    "of", "onto", "upon", "within", "without", "across", "around",
    "behind", "beside", "beyond", "throughout", "toward", "towards",
    # quantifiers / determiners / degree
    "all", "any", "each", "every", "few", "more", "most", "other",
    "another", "some", "such", "no", "not", "only", "own", "same",
    "too", "very", "just", "now", "then", "here", "there",
    "much", "many", "several", "enough",
    # negation / interrogative
    "never", "none", "nothing", "nobody", "nowhere",
    "why", "how", "ever", "once",
    # particles / misc closed-class
    "s", "t", "d", "ll", "re", "ve", "m",           # contraction fragments
    "yes", "no", "ok", "okay",
    # articles-of-light verbs commonly treated as function-adjacent
    "said", "says", "say", "saying",
    "got", "get", "getting", "gets",
    "goes", "going", "went", "gone", "go",
    "came", "come", "comes", "coming",
    "made", "make", "makes", "making",
    "let", "lets", "letting",
    # extra high-frequency glue
    "one", "two", "three", "first", "last", "next",
    "also", "however", "therefore", "thus", "still", "even",
    "already", "always", "sometimes", "often", "again",
    "like", "unlike",
    # dialogue interjections / discourse markers. These get capitalized at
    # dialogue start ("Hi Joey!") and get falsely confirmed as entities
    # without this filter.
    "hi", "hello", "hey", "yeah", "yep", "nope", "well", "sure",
    "really", "oh", "uh", "um", "ah", "eh", "wow", "alright",
    "thanks", "please", "sorry", "hmm", "bye", "goodbye",
    "cool", "nice", "great", "maybe", "perhaps", "indeed",
})


# ---------------------------------------------------------------------------
# tokenization helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z']+")

# Unicode curly quotes normalize to ASCII so contractions like "couldn't"
# render correctly no matter which character the model emitted. Without
# this, "couldn't" becomes the two tokens "couldn" + "t", and "couldn"
# gets flagged as a made-up word by the speller.
_QUOTE_TRANSLATE = str.maketrans({
    "‘": "'",  # LEFT SINGLE QUOTATION MARK
    "’": "'",  # RIGHT SINGLE QUOTATION MARK
    "“": '"',  # LEFT DOUBLE QUOTATION MARK
    "”": '"',  # RIGHT DOUBLE QUOTATION MARK
    "′": "'",  # PRIME
})


def _normalize_quotes(text: str) -> str:
    return text.translate(_QUOTE_TRANSLATE)


def to_words(text: str) -> List[str]:
    """Lowercase word sequence. Used for all word-level n-gram metrics.

    Curly quotes are normalized to ASCII first. Internal apostrophes
    (contractions, possessives) are kept. Leading/trailing apostrophes are
    stripped — those are typically dialogue-quote boundaries, not part of
    the word itself, and leaving them in makes the nonword-rate speller
    falsely flag ordinary words like 'is or louise'.
    """
    text = _normalize_quotes(text)
    words = []
    for m in _WORD_RE.finditer(text):
        w = m.group(0).lower().strip("'")
        if w:
            words.append(w)
    return words


# ---------------------------------------------------------------------------
# repetition / diversity metrics
# ---------------------------------------------------------------------------

def seq_rep_n(words: Sequence[str], n: int) -> float:
    """Fraction of n-grams that are duplicates of an earlier n-gram.

    0.0 = no repetition. Healthy long-form text is typically < 0.05 for n=4.
    Degenerate / looping text climbs above ~0.15.
    """
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return 1.0 - (unique / total)


def distinct_n(words: Sequence[str], n: int) -> float:
    """Unique-to-total n-gram ratio. Complement of seq_rep_n's perspective.

    1.0 = every n-gram novel. Collapses toward 0 as the text becomes repetitive.
    """
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def compression_ratio(text: str) -> float:
    """gzip(compressed) / raw, on utf-8 bytes.

    Repetitive / low-entropy text compresses well (small ratio). Natural prose
    sits around 0.4–0.55 for this token budget.
    """
    if not text:
        return 0.0
    raw = text.encode("utf-8")
    compressed = gzip.compress(raw)
    return len(compressed) / len(raw)


def mattr(words: Sequence[str], window: int = 50) -> float:
    """Moving-Average Type-Token Ratio. Length-insensitive vocabulary richness.

    For each length-`window` sliding window, compute unique/total; average
    across all windows. Falls back to plain TTR if the generation is shorter
    than the window.
    """
    if len(words) == 0:
        return 0.0
    if len(words) < window:
        return len(set(words)) / len(words)
    ratios = []
    for i in range(len(words) - window + 1):
        chunk = words[i:i + window]
        ratios.append(len(set(chunk)) / window)
    return sum(ratios) / len(ratios)


# ---------------------------------------------------------------------------
# entity metrics
# ---------------------------------------------------------------------------

_CAP_WORD_RE = re.compile(r"\b[A-Z][A-Za-z'\-]{1,}\b")
_SENT_TERMS = set(".!?")


_CONTRACTION_SUFFIXES = ("'s", "'d", "'m", "'ll", "'ve", "'re", "'t")


def _normalize_entity(word: str) -> str:
    """Collapse surface inflections so the same named thing matches itself,
    and contractions of "I" don't masquerade as proper nouns.

    - Strip trailing possessive 's  (Whitfield's -> Whitfield)
    - Strip other contraction tails (I'd -> I, I'm -> I, I've -> I, ...)
    - Strip trailing lone apostrophe (Thomas' -> Thomas)
    Plural proper nouns (Whitfields, Charleses) are left alone — stripping
    trailing `s` would incorrectly rewrite real names like Charles.
    """
    for suf in _CONTRACTION_SUFFIXES:
        if word.endswith(suf):
            return word[:-len(suf)]
    if word.endswith("'"):
        return word[:-1]
    return word


def _extract_cap_runs(text: str) -> List[tuple]:
    """Find maximal runs of consecutive capitalized words separated only by
    whitespace, collapsing each run into a single entity string.

    Returns list of (entity, start_pos, end_pos). "Air Force" becomes one
    entity, as does "New York City". "Dr. Ramos" stays split (the period
    blocks the merge). Consecutive cap words across sentence terminators
    ("store. Later") won't merge either (period blocks).

    Interjection / function-word caps (Hi, Yeah, Well, Thanks, ...) are
    filtered by STOPWORDS membership before merging — otherwise dialogue
    like "Hi, Wendy" gets confirmed as an entity and then merged with names.
    """
    raw = []
    for m in _CAP_WORD_RE.finditer(text):
        w = _normalize_entity(m.group(0))
        if not w or w == "I":
            continue
        if w.lower() in STOPWORDS:
            continue
        raw.append((w, m.start(), m.end()))

    merged = []
    i = 0
    while i < len(raw):
        parts = [raw[i][0]]
        start = raw[i][1]
        end = raw[i][2]
        j = i + 1
        while j < len(raw):
            gap = text[end:raw[j][1]]
            if gap and all(c in " \t" for c in gap):
                parts.append(raw[j][0])
                end = raw[j][2]
                j += 1
            else:
                break
        merged.append((" ".join(parts), start, end))
        i = j
    return merged


def _extract_entities(text: str) -> List[tuple]:
    """Return list of (entity, char_offset).

    Two-pass: a capitalized word (or multi-word cap run like "Air Force")
    is counted as an entity if it appears at least once *not* at a sentence
    start (that confirms it's a proper noun, not just a sentence-initial
    cap). Once confirmed, every occurrence is counted — including sentence-
    initial ones — so protagonist mentions aren't systematically discarded.
    Surface inflections (possessive 's) are normalized so the same name is
    one entity. Multi-word proper nouns collapse into a single entity.
    """
    text = _normalize_quotes(text)
    runs = _extract_cap_runs(text)

    candidates = []  # (entity, start_pos, is_sentence_initial)
    for word, start, _ in runs:
        j = start - 1
        while j >= 0 and text[j] in " \t\n\r\"'`([{":
            j -= 1
        is_sentence_initial = (j < 0) or (text[j] in _SENT_TERMS)
        candidates.append((word, start, is_sentence_initial))

    confirmed = {w for w, _, sent_init in candidates if not sent_init}
    return [(w, pos) for w, pos, _ in candidates if w in confirmed]


def _prompt_entities(text: str) -> set:
    """Permissive entity extraction for prompts.

    Prompts are short; proper-noun introductions are frequently sentence-
    initial (e.g. "Elena pushed open the door..."), so the strict two-pass
    rule used for generations would miss them. Here we trust the
    capitalization: every capitalized word (or multi-word cap run like
    "Air Force") counts as a given.

    Returns BOTH the merged forms and the individual components, so that if
    the prompt names "Detective Sarah Chen" and the generation refers to
    just "Chen", the component match still counts "Chen" as a given.
    """
    text = _normalize_quotes(text)
    out = set()
    for word, _, _ in _extract_cap_runs(text):
        out.add(word)
        for part in word.split():
            out.add(part)
    return out


def entity_persistence(text: str) -> float:
    """total_mentions / unique_entities.

    Coherent narrative: > 3.0 (a handful of characters named repeatedly).
    Incoherent / name-spray: → 1.0 (every proper noun mentioned once).
    Returns 0.0 if no entities are found.
    """
    ents = [e for e, _ in _extract_entities(text)]
    if not ents:
        return 0.0
    unique = len(set(ents))
    return len(ents) / unique


def cross_span_entity(text: str, prompt_text: Optional[str] = None) -> float:
    """Fraction of unique entities that appear in both the first-third and
    last-third character spans of the generation.

    1.0 = every entity carries across the full span (strong continuity).
    0.0 = no entity appears in both ends (characters don't persist).
    Returns 0.0 if no entities are found.

    When `prompt_text` is provided, prompt-given entities are treated as if
    they were present in the first third. This credits the model for
    referencing a prompt character late in the generation even if that
    character only appears once (which would otherwise fail the two-region
    intersection check). Aggregate cliff signal is preserved because the
    broken model still won't carry prompt characters into the last third.
    """
    ents = _extract_entities(text)
    if not ents:
        return 0.0
    n = len(text)
    if n < 3:
        return 0.0
    first_end = n // 3
    last_start = 2 * n // 3
    first_set = {e for e, pos in ents if pos < first_end}
    last_set = {e for e, pos in ents if pos >= last_start}
    if prompt_text is not None:
        first_set = first_set | _prompt_entities(prompt_text)
    unique = {e for e, _ in ents}
    both = first_set & last_set
    return len(both) / len(unique)


# ---------------------------------------------------------------------------
# phase 1.5: metrics targeting "semantic drift without repetition"
#   - model introduces new entities without calling back
#   - vocabulary stays diverse (so seq_rep / mattr look healthy)
#   - characters don't persist across spans
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def _split_sentences(text: str) -> List[str]:
    """Cheap sentence splitter — good enough for metric aggregation over
    generated prose. Not meant to handle abbreviations perfectly."""
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return parts


def _entities_per_sentence(text: str) -> List[set]:
    """Return a list of entity sets, one per sentence, using the same
    confirmed-entity rules as _extract_entities."""
    ents_with_pos = _extract_entities(text)
    sentences = _split_sentences(text)
    # Compute character offsets of each sentence start in the original text
    # so we can assign each entity to a sentence.
    sent_ranges = []
    cursor = 0
    for s in sentences:
        start = text.find(s, cursor)
        if start < 0:
            start = cursor
        end = start + len(s)
        sent_ranges.append((start, end))
        cursor = end
    per_sent = [set() for _ in sentences]
    for word, pos in ents_with_pos:
        # Assign to first sentence whose span covers `pos`.
        for i, (a, b) in enumerate(sent_ranges):
            if a <= pos < b:
                per_sent[i].add(word)
                break
    return per_sent


def new_entities_introduced(gen_text: str, prompt_text: Optional[str] = None) -> int:
    """Count of distinct entities in the generation that were NOT present
    in the prompt.

    Captures the "name spray" failure mode: a drifting model introduces
    new characters constantly without calling back. Each new entity is
    counted once regardless of how often it appears.

    Coherent narrative: 0–4 (model reuses prompt characters, adds a few
    NPCs). Drifting model: 8+ (a new character every paragraph).

    If `prompt_text` is None, no subtraction is performed — this becomes a
    plain unique-entity count.

    Coverage check is component-wise: a gen entity is considered "given" if
    all its whitespace-separated components appear in the prompt's
    entity set (merged or individual). So "Sarah Chen" in the generation
    counts as given when the prompt named "Detective Sarah Chen".
    """
    gen_ents = {e for e, _ in _extract_entities(gen_text)}
    if prompt_text is None:
        return len(gen_ents)
    givens = _prompt_entities(prompt_text)
    new = set()
    for e in gen_ents:
        if e in givens:
            continue
        if all(part in givens for part in e.split()):
            continue
        new.add(e)
    return len(new)


def novel_entity_rate(text: str) -> float:
    """Deprecated: superseded by new_entities_introduced. Retained for
    backward compatibility with older analysis scripts.

    Returns unique entities per 100 generated words. The metric has a
    prompt-dependent baseline (e.g. historical prompts naturally produce
    many names) which new_entities_introduced removes.
    """
    words = to_words(text)
    if not words:
        return 0.0
    unique = len(set(e for e, _ in _extract_entities(text)))
    return 100.0 * unique / len(words)


def entity_chain_length(text: str) -> int:
    """Longest run of consecutive sentences that share at least one entity
    with the previous sentence.

    Coherent narrative: long chains (characters persist across many sentences).
    Drifting model: very short chains (each sentence introduces fresh entities).
    Returns 0 if no multi-sentence chain exists.
    """
    per_sent = _entities_per_sentence(text)
    if len(per_sent) < 2:
        return 0
    best = 0
    current = 0
    for i in range(1, len(per_sent)):
        if per_sent[i] and per_sent[i - 1] and (per_sent[i] & per_sent[i - 1]):
            current += 1
            best = max(best, current + 1)  # +1 to count both endpoints
        else:
            current = 0
    return best


def sentence_entity_overlap(text: str) -> float:
    """Mean Jaccard overlap of entity sets between adjacent sentences.

    Coherent narrative: 0.3–0.6 (many sentences share at least one named
    character with their neighbor). Drifting model: near 0 (each sentence
    has entirely fresh entities).
    Returns 0.0 if there are fewer than two entity-bearing sentences.
    """
    per_sent = _entities_per_sentence(text)
    if len(per_sent) < 2:
        return 0.0
    scores = []
    for i in range(1, len(per_sent)):
        a, b = per_sent[i - 1], per_sent[i]
        if not a and not b:
            continue
        union = a | b
        if not union:
            continue
        scores.append(len(a & b) / len(union))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# phase 1.5: made-up word detection
# ---------------------------------------------------------------------------
#
# Dictionary backend: wordfreq's 'large' English wordlist (~320k words drawn
# from Wikipedia, subtitles, news, books, web, Twitter, Reddit). Empirically
# recognizes rare-but-real words that pyspellchecker misses (fete, loch, lich,
# buzzy, dwarven, midafternoon, ...) while still flagging the model's genuine
# hallucinations (arouseing, cordill, flornate, verbation, ...).

_KNOWN_WORDS = None  # lazy singleton (frozenset)


def _get_known_words():
    """Lazy-load the English wordfreq 'large' list as a frozenset for O(1)
    membership tests. Raises ModuleNotFoundError if wordfreq is missing."""
    global _KNOWN_WORDS
    if _KNOWN_WORDS is not None:
        return _KNOWN_WORDS
    try:
        import wordfreq
    except ImportError as e:
        raise ModuleNotFoundError(
            "coherence_metrics.nonword_rate requires the 'wordfreq' package. "
            "Install it with: pip install wordfreq"
        ) from e
    # top_n with a very large N returns the entire 'large' list.
    words = wordfreq.top_n_list("en", 2_000_000, wordlist="large")
    _KNOWN_WORDS = frozenset(words)
    return _KNOWN_WORDS


def nonword_rate(text: str) -> float:
    """Fraction of alphabetic word tokens that are NOT in the English
    dictionary.

    Targets the "making up words" failure mode: a drifting model may emit
    phonetically plausible but non-existent tokens ("verbation", "flornate").
    Coherent narrative: < 0.02 (mostly just real rare words or proper-noun
    casing edge cases). Drifting model: rises markedly.

    Filters:
      - Lowercased word form
      - Drops proper-noun candidates (any word that appears capitalized
        mid-sentence at least once in `text`)
      - Drops 1–2 letter words (mostly contraction fragments)

    Raises ModuleNotFoundError if wordfreq is not installed.
    """
    known = _get_known_words()

    # Proper-noun exclusion set — any confirmed entity (lowercased) should
    # not count as a made-up word even if it's not in the dict.
    proper_nouns = {w.lower() for w, _ in _extract_entities(text)}

    words = to_words(text)
    candidates = [w for w in words if len(w) >= 3 and w not in proper_nouns]
    if not candidates:
        return 0.0

    unknown_count = sum(1 for w in candidates if w not in known)
    return unknown_count / len(candidates)


# ---------------------------------------------------------------------------
# intrinsic (logit-based) metric
# ---------------------------------------------------------------------------

def entropy_ratio(
    token_strings: Sequence[str],
    per_token_entropy: Sequence[float],
) -> Optional[float]:
    """mean(entropy on content tokens) / mean(entropy on function tokens).

    `token_strings` is the decoded-in-isolation form of each generated subword
    token, aligned 1:1 with `per_token_entropy`. A subword is classified by:
      - strip whitespace, lowercase → key
      - key in STOPWORDS            → function
      - key is alphabetic, len >= 3 → content
      - otherwise                   → skipped (punctuation, numerics, fragments)

    Healthy model: ratio around or under 1.0 (at least as confident on content
    as on function words). Degraded model: ratio >> 1.0 (function words stay
    easy, content discrimination collapses).

    Returns None if either class has no samples.
    """
    if len(token_strings) != len(per_token_entropy):
        raise ValueError(
            f"token_strings ({len(token_strings)}) and per_token_entropy "
            f"({len(per_token_entropy)}) must be the same length"
        )

    content_ents = []
    function_ents = []
    for tok, ent in zip(token_strings, per_token_entropy):
        key = tok.strip().lower()
        if not key:
            continue
        if key in STOPWORDS:
            function_ents.append(ent)
        elif key.isalpha() and len(key) >= 3:
            content_ents.append(ent)
        # else: punctuation, partial-word bpe pieces, numerics → skip

    if not content_ents or not function_ents:
        return None
    return (sum(content_ents) / len(content_ents)) / (
        sum(function_ents) / len(function_ents)
    )


# ---------------------------------------------------------------------------
# top-level aggregator
# ---------------------------------------------------------------------------

def compute_all(
    text: str,
    prompt_text: Optional[str] = None,
    token_strings: Optional[Sequence[str]] = None,
    per_token_entropy: Optional[Sequence[float]] = None,
) -> dict:
    """Compute every metric on a single generated sample.

    `prompt_text` is used by `new_entities_introduced` to subtract the
    prompt's given characters from the generation's entities. Scoring is
    less accurate without it.

    `token_strings` and `per_token_entropy` are optional; the entropy_ratio
    metric is returned as None if they're missing.
    """
    words = to_words(text)
    out = {
        "n_words": len(words),
        "n_chars": len(text),
        # phase 1: repetition / diversity (Holtzman-style)
        "seq_rep_2": seq_rep_n(words, 2),
        "seq_rep_3": seq_rep_n(words, 3),
        "seq_rep_4": seq_rep_n(words, 4),
        "distinct_1": distinct_n(words, 1),
        "distinct_2": distinct_n(words, 2),
        "distinct_3": distinct_n(words, 3),
        "compression": compression_ratio(text),
        "mattr": mattr(words, window=50),
        # phase 1: entity continuity
        "entity_persist": entity_persistence(text),
        "cross_span_entity": cross_span_entity(text, prompt_text),
        # phase 1.5: semantic drift (designed after Dreadnought v1 analysis)
        "new_entities_introduced": new_entities_introduced(text, prompt_text),
        "entity_chain_length": entity_chain_length(text),
        "sentence_entity_overlap": sentence_entity_overlap(text),
        "nonword_rate": nonword_rate(text),
    }
    if token_strings is not None and per_token_entropy is not None:
        out["entropy_ratio"] = entropy_ratio(token_strings, per_token_entropy)
    else:
        out["entropy_ratio"] = None
    return out


def aggregate(per_prompt: List[dict]) -> dict:
    """Aggregate each metric across a list of per-prompt results.

    Most metrics are averaged. `new_entities_introduced` is aggregated as
    median (primary dashboard signal) plus mean, min, and max as secondary
    stats — median is robust to prompts whose baselines are naturally
    name-heavy (e.g. the historical_grounded prompt).

    None values are excluded from their respective aggregates.
    """
    if not per_prompt:
        return {}
    keys = [k for k in per_prompt[0].keys() if k not in ("n_words", "n_chars")]
    agg = {}
    for k in keys:
        vals = [p[k] for p in per_prompt if p.get(k) is not None]
        if k == "new_entities_introduced":
            # Report median/mean/min/max separately.
            if vals:
                s = sorted(vals)
                mid = len(s) // 2
                median = s[mid] if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) / 2
                agg["new_entities_introduced_median"] = median
                agg["new_entities_introduced_mean"] = sum(vals) / len(vals)
                agg["new_entities_introduced_min"] = min(vals)
                agg["new_entities_introduced_max"] = max(vals)
            else:
                agg["new_entities_introduced_median"] = None
                agg["new_entities_introduced_mean"] = None
                agg["new_entities_introduced_min"] = None
                agg["new_entities_introduced_max"] = None
        else:
            agg[k] = (sum(vals) / len(vals)) if vals else None
    # Also record total word / char counts so we can sanity-check generation length.
    agg["total_words"] = sum(p.get("n_words", 0) for p in per_prompt)
    agg["total_chars"] = sum(p.get("n_chars", 0) for p in per_prompt)
    return agg
