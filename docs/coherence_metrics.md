# coherence_metrics.py
> Pure-Python library of text-degeneration metrics computed on already-generated LLM output — no model, no GPU, no checkpoint load.

## What it does
Scores a string of generated text for degeneration signatures: repetition/looping, vocabulary collapse, name-spray and semantic drift (entities that don't persist across the span), made-up words, and (optionally) an intrinsic entropy-ratio from per-token entropies emitted during generation. It is a stateless metrics library you import — feed it generated text (plus the prompt and, optionally, aligned per-token data) and get back a dict of scores. Reach for it when you have generation logs from a run and want to quantify "is this output coherent or has it gone off the rails" without re-running the model. Two metric phases are baked in: phase 1 (Holtzman/Li-style repetition + diversity + entity continuity) and phase 1.5 (semantic-drift-without-repetition: new-entity introduction, entity chains, sentence overlap, nonword rate).

## Usage
There is **no CLI** — the file has no `argparse`, no `main()`, and no `if __name__ == "__main__"` block. It is an importable library. Call `compute_all()` on one sample, or `aggregate()` over many.

```python
import coherence_metrics as cm

# one generated sample (prompt optional but improves entity scoring)
result = cm.compute_all(
    text=generation_str,
    prompt_text=prompt_str,            # optional; used to subtract "given" entities
    token_strings=tok_strs,            # optional; aligned 1:1 with entropies
    per_token_entropy=entropies,       # optional; enables entropy_ratio
)

# aggregate a list of per-sample dicts into run-level stats
summary = cm.aggregate([cm.compute_all(t, p) for t, p in samples])

# individual metrics are also public, e.g.:
cm.seq_rep_n(cm.to_words(text), 4)
cm.nonword_rate(text)                  # requires the `wordfreq` package
```

## Key metrics (returned by `compute_all`)
| key | what it measures | rough healthy range (per source) |
| --- | --- | --- |
| `seq_rep_2/3/4` | fraction of duplicate n-grams (repetition/looping) | n=4 < 0.05 |
| `distinct_1/2/3` | unique/total n-gram ratio (diversity) | near 1.0 |
| `compression` | gzip ratio on utf-8 bytes (low = repetitive) | ~0.4–0.55 |
| `mattr` | moving-avg type-token ratio, window=50 (length-insensitive richness) | — |
| `entity_persist` | total mentions / unique entities | > 3.0 coherent; → 1.0 name-spray |
| `cross_span_entity` | entities appearing in both first- and last-third spans | 1.0 = full continuity |
| `new_entities_introduced` | distinct gen entities not in the prompt | 0–4 coherent; 8+ drifting |
| `entity_chain_length` | longest run of consecutive sentences sharing an entity | longer = more coherent |
| `sentence_entity_overlap` | mean Jaccard of entity sets between adjacent sentences | 0.3–0.6 coherent |
| `nonword_rate` | fraction of alpha tokens not in the English dictionary | < 0.02 coherent |
| `entropy_ratio` | mean entropy on content tokens / on function tokens | ≤ ~1.0 healthy; >> 1.0 degraded |

`aggregate()` averages most metrics across samples; `new_entities_introduced` is reported as `_median` (primary), plus `_mean`/`_min`/`_max`. It also sums `total_words`/`total_chars`. `None` values are excluded from their aggregates.

## Notes
- **`nonword_rate` requires the `wordfreq` package** (uses the `large` English wordlist, lazily loaded once into a frozenset). If `wordfreq` is missing it raises `ModuleNotFoundError`. All other metrics are stdlib-only (`gzip`, `re`, `math`).
- **`entropy_ratio` needs aligned arrays**: `token_strings` are the decoded-in-isolation subword tokens, 1:1 with `per_token_entropy` (the model's own per-token entropies captured during generation). Mismatched lengths raise `ValueError`; returns `None` if either content/function class is empty. If you don't pass these, `compute_all` sets `entropy_ratio` to `None`.
- **Entity detection is heuristic** (capitalization + a ~200-word STOPWORDS filter, two-pass "confirm a cap word is a proper noun only if it appears non-sentence-initially"). Curly quotes are normalized to ASCII; possessive/contraction tails are stripped. Multi-word cap runs ("New York City") merge into one entity. Prompts use a more permissive extraction (`_prompt_entities`) that trusts capitalization, since proper nouns are often sentence-initial there.
- Thresholds quoted in docstrings are calibration heuristics, not hard validated cutoffs.
- `novel_entity_rate()` is **deprecated**, superseded by `new_entities_introduced`; kept for backward compatibility.
- Comments reference prior analyses ("Dreadnought v1"); this looks like part of a generation-quality / coherence evaluation effort. No checkpoint or run-dir dependency — it operates purely on logged text.
