# recompute_coherence_metrics.py
#
# Re-run coherence_metrics.compute_all over an existing coherence_log.jsonl
# without regenerating any text. This is how we iterate on the metric
# library cheaply — stored per_prompt[*].text fields are frozen inputs.
#
# Usage:
#   python recompute_coherence_metrics.py <input.jsonl> [<output.jsonl>]
#
# If no output path is given, writes to <input>.recomputed.jsonl.
#
# Notes:
#   - entropy_ratio cannot be recomputed (needs per-token entropies from the
#     model, which aren't stored). Its value is preserved from the input
#     record as-is.
#   - Only the "metrics" fields inside per_prompt and the top-level
#     "aggregate" object are rewritten. Everything else (step, tokens,
#     generation config, per_prompt.text, etc.) is passed through unchanged.

import json
import sys
from pathlib import Path

import coherence_metrics as cm
from redact_coherence_log import redact_file


DEFAULT_PROMPTS_PATH = Path(__file__).parent / "coherence_prompts.json"


def load_prompt_map(prompts_path: Path) -> dict:
    """Map prompt id -> prompt text, for subtracting givens in
    new_entities_introduced."""
    if not prompts_path.is_file():
        return {}
    with prompts_path.open("r", encoding="utf-8") as f:
        bank = json.load(f)
    return {p["id"]: p["text"] for p in bank.get("prompts", [])}


def recompute_record(record: dict, prompt_map: dict) -> dict:
    per_prompt = record.get("per_prompt", [])
    for entry in per_prompt:
        text = entry.get("text", "")
        if not isinstance(text, str):
            continue
        # Preserve the old entropy_ratio so we don't lose that signal.
        old_entropy = None
        old_metrics = entry.get("metrics", {})
        if isinstance(old_metrics, dict):
            old_entropy = old_metrics.get("entropy_ratio")
        prompt_text = prompt_map.get(entry.get("id"))
        new_metrics = cm.compute_all(text, prompt_text=prompt_text)
        new_metrics["entropy_ratio"] = old_entropy
        entry["metrics"] = new_metrics
    record["aggregate"] = cm.aggregate([p["metrics"] for p in per_prompt])
    return record


def main():
    if len(sys.argv) < 2:
        print("usage: python recompute_coherence_metrics.py <input.jsonl> [<output.jsonl>] [--prompts <path>]")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    prompts_path = DEFAULT_PROMPTS_PATH
    for i, a in enumerate(sys.argv):
        if a == "--prompts" and i + 1 < len(sys.argv):
            prompts_path = Path(sys.argv[i + 1])

    in_path = Path(args[0])
    if not in_path.is_file():
        print(f"ERROR: not a file: {in_path}", file=sys.stderr)
        sys.exit(1)

    if len(args) >= 2:
        out_path = Path(args[1])
    else:
        out_path = in_path.parent / (in_path.stem + ".recomputed.jsonl")

    prompt_map = load_prompt_map(prompts_path)
    if not prompt_map:
        print(f"  WARN: no prompts loaded from {prompts_path} — new_entities_introduced will be un-subtracted", file=sys.stderr)

    n_recs = 0
    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARN: skipping malformed line: {e}", file=sys.stderr)
                continue
            record = recompute_record(record, prompt_map)
            fout.write(json.dumps(record) + "\n")
            n_recs += 1
            print(f"  recomputed step {record.get('step')}")

    print(f"\nDone. {n_recs} record(s) written to {out_path}")

    # Auto-redact alongside the recomputed file so downstream agents always
    # see the latest values without manual intervention.
    try:
        redacted_path, n_red, _ = redact_file(out_path)
        print(f"Redacted log: {redacted_path}  ({n_red} record(s))")
    except Exception as e:
        print(f"  WARN: failed to write redacted log: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
