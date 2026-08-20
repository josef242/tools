# recompute_coherence_metrics.py
> Re-runs the coherence-metric library over an existing `coherence_log.jsonl` using the frozen, stored generation text — no model load, no regeneration.

## What it does
Iterates over a JSONL coherence log (one record per training step), re-computes every metric in `coherence_metrics.compute_all` on each stored `per_prompt[*].text`, and rewrites the per-prompt `metrics` block and the top-level `aggregate` object. Everything else in each record (`step`, token counts, generation config, the raw `text` fields, etc.) is passed through unchanged. This is the cheap iteration loop for the coherence-metric library: the generated text is a frozen input, so you can evolve metric definitions and rescore historical logs without re-running generation.

It sits alongside two siblings it imports directly: `coherence_metrics.py` (the metric library: `compute_all` / `aggregate`) and `redact_coherence_log.py` (text stripping). It also reads a prompt bank, `coherence_prompts.json`.

## Usage
```bash
# Recompute in place to a sibling <input>.recomputed.jsonl
python recompute_coherence_metrics.py coherence_log.jsonl

# Explicit output path
python recompute_coherence_metrics.py coherence_log.jsonl coherence_log.new.jsonl

# Override the prompt bank used to subtract prompt-given entities
python recompute_coherence_metrics.py coherence_log.jsonl --prompts my_prompts.json
```

This is hand-rolled `sys.argv` parsing (no `argparse`). Positional args are everything not starting with `--`; the first is the input, an optional second is the output. If no output is given it writes to `<input_stem>.recomputed.jsonl` in the input's directory.

## Key arguments

| arg | default | meaning |
|-----|---------|---------|
| `<input.jsonl>` | required | Coherence log to rescore (one JSON record per line). |
| `<output.jsonl>` | `<input_stem>.recomputed.jsonl` (same dir) | Where the rewritten log is written. |
| `--prompts <path>` | `coherence_prompts.json` next to the script | Prompt bank mapping prompt `id` -> prompt text, used to subtract prompt-given entities in `new_entities_introduced`. |

## Notes
- `entropy_ratio` cannot be recomputed: it needs per-token entropies from the model, which aren't stored in the log. The old value is read from the input record's `metrics` and preserved as-is; if absent it ends up `None`.
- The prompt bank is used only to subtract "given" characters/entities for the `new_entities_introduced` metric. If no prompts load (missing file or empty), it warns and proceeds with that metric un-subtracted (less accurate, not fatal).
- Per-prompt entries are matched to prompts by `entry["id"]` against the bank's `prompts[*].id`. Entries whose `text` is not a string are skipped.
- Malformed JSON lines are warned and skipped; blank lines are ignored.
- `aggregate` averages most metrics and reports `new_entities_introduced` as separate median/mean/min/max keys; `None` values are excluded from each aggregate. `n_words`/`n_chars` are excluded from per-metric aggregation but summed into `total_words`/`total_chars`.
- After writing the recomputed file, it auto-calls `redact_file` from `redact_coherence_log.py` to emit a `<...>.redacted.jsonl` sibling with all `per_prompt[*].text` replaced by `<redacted>` (the generations are from an AO3-trained model and may contain flagged material). A redaction failure is caught and warned, not fatal.
- Must be run from a location where `coherence_metrics` and `redact_coherence_log` are importable (i.e. the `tools/` directory) and where the conda training env's deps for the metric library are available.
