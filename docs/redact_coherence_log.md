# redact_coherence_log.py
> Strips raw generated text out of a `coherence_log.jsonl` so the file can be shared without exposing AO3-trained generations to content filters.

## What it does
This is a small, standalone sanitizer for coherence-metric logs. It reads one or
more `coherence_log.jsonl` files and, for every record, replaces each
`per_prompt[*].text` field with the literal string `"<redacted>"` while leaving
all metrics, step/token counts, and other metadata untouched. You reach for it
when you want to hand off or archive a coherence log but the underlying model was
trained on unfiltered AO3 text, so the raw generations may contain flagged
material. It does no model load and no analysis — it is purely a field-level
redaction pass over JSONL.

## Usage
No argparse. It is a fixed-behavior CLI driven by positional `sys.argv` paths.
It accepts one or more input JSONL files; each produces a sibling
`*.redacted.jsonl`.

```bash
# single file -> coherence_log.redacted.jsonl in the same directory
python redact_coherence_log.py coherence_log.jsonl

# multiple files in one pass
python redact_coherence_log.py run_a/coherence_log.jsonl run_b/coherence_log.jsonl
```

With no arguments it prints a usage line and exits with status 1.

## Notes
- **Output naming**: for input `X.jsonl` it writes `X.redacted.jsonl` in the
  same parent directory (built as `parent / (stem + ".redacted.jsonl")`).
- **What gets redacted**: only `record["per_prompt"]` entries that are dicts
  containing a `"text"` key. Each `text` is replaced with `"<redacted>"`, and a
  new field `text_redacted_original_chars` is added recording the original
  character length (0 if the original was not a string). All other fields pass
  through unchanged.
- **Robustness**: blank lines are skipped; malformed JSON lines are skipped with
  a `WARN` to stderr and processing continues. Non-file arguments are reported as
  an `ERROR` to stderr and skipped.
- **Per-file summary**: prints the input path, the output path, and a count of
  records processed plus text fields redacted.
- **Dependencies**: standard library only (`json`, `sys`, `pathlib`). No
  checkpoint, GPU, or special conda env required. Reads/writes UTF-8.
- **Caveat**: redaction is keyed on the `per_prompt[*].text` schema. If a
  coherence log stores generated text under a different field or nesting, that
  text would not be caught — verify the schema before trusting a redacted file
  for sharing.
