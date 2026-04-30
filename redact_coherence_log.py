# redact_coherence_log.py
#
# Strip raw generated text from a coherence_log.jsonl so it can be shared
# without risk of triggering content filters (the model was trained on AO3
# and some generations may contain flagged material).
#
# Usage:
#   python redact_coherence_log.py <input.jsonl> [<input2.jsonl> ...]
#
# For each input, writes a sibling file with the ".redacted.jsonl" suffix
# (e.g. coherence_log.jsonl -> coherence_log.redacted.jsonl). The metrics,
# step/token counts, and all other metadata are preserved exactly — only the
# per_prompt[*].text fields are replaced with "<redacted>".

import json
import sys
from pathlib import Path


REDACTION = "<redacted>"


def redact_record(record: dict) -> dict:
    per_prompt = record.get("per_prompt")
    if isinstance(per_prompt, list):
        for entry in per_prompt:
            if "text" in entry:
                original_len = len(entry["text"]) if isinstance(entry["text"], str) else 0
                entry["text"] = REDACTION
                entry["text_redacted_original_chars"] = original_len
    return record


def redact_file(in_path: Path) -> Path:
    out_path = in_path.with_suffix("")  # drop .jsonl
    # If the stem already ends with something weird, just append:
    out_path = in_path.parent / (in_path.stem + ".redacted.jsonl")

    n_in = 0
    n_redacted_texts = 0
    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARN: skipping malformed line {n_in}: {e}", file=sys.stderr)
                continue
            before = sum(
                1 for p in record.get("per_prompt", [])
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            )
            redact_record(record)
            n_redacted_texts += before
            fout.write(json.dumps(record) + "\n")

    return out_path, n_in, n_redacted_texts


def main():
    if len(sys.argv) < 2:
        print("usage: python redact_coherence_log.py <input.jsonl> [<input2.jsonl> ...]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        in_path = Path(arg)
        if not in_path.is_file():
            print(f"  ERROR: not a file: {in_path}", file=sys.stderr)
            continue
        out_path, n_records, n_texts = redact_file(in_path)
        print(f"  {in_path}")
        print(f"    -> {out_path}")
        print(f"    {n_records} record(s), {n_texts} text field(s) redacted")


if __name__ == "__main__":
    main()
