# eval.py
> Runs standard LLM benchmarks (HellaSwag, MMLU, ARC-Easy/Challenge, GSM8K, HumanEval) on a trained model checkpoint and writes a JSON results file.

## What it does
Loads a single model checkpoint and evaluates it against a battery of public
benchmarks, reporting per-test accuracy plus an overall summary. The multiple-choice
tests (HellaSwag, MMLU, ARC) use NLL-based scoring — encode the prompt plus each
candidate answer, compute the loss on the answer span, and pick the lowest-loss
choice. GSM8K and HumanEval are generative: GSM8K few-shot answer extraction and
HumanEval generates code completions that are executed in a sandboxed subprocess.
Reach for this when you have a checkpoint and want comparable benchmark numbers.

It is a thin orchestration layer over `generate_neo.py` and `neo_common.py` (`nc`):
it reuses `resolve_model_path`, `get_checkpoint_info`, `load_model_and_tokenizer`,
`score_hellaswag_batch`, `pad_and_stack`, `get_batch_loss`, `test_gsm8k`, and
`stream_generate_kv`. The MMLU and ARC loops are implemented locally so they can
return correct/total counts for the JSON output.

## Usage
```bash
# Run a few NLL-based benchmarks on the latest checkpoint in a run dir
python eval.py <checkpoint_path> --test hellaswag --test mmlu --test arc-easy

# Run everything, write to a named results file
python eval.py <checkpoint_path> --test all --output my_results.json

# Pick a GPU and use full precision
python eval.py <checkpoint_path> --test gsm8k --gpu 0 --full
```
`<checkpoint_path>` may be a `.pt` file or a run directory (latest checkpoint is
auto-selected via `resolve_model_path`). At least one `--test` is required; with
none, the tool prints the available tests and exits with code 1.

## Key arguments
| flag | default | meaning |
|------|---------|---------|
| `checkpoint` (positional) | — | Path to checkpoint `.pt` file or run directory (auto-selects latest) |
| `--test` | none (required) | Test to run; repeatable. Choices: `hellaswag`, `mmlu`, `arc-easy`, `arc-challenge`, `gsm8k`, `humaneval`, or `all` |
| `--output` | `<parent_dir>_step<N>_results.json` | Output JSON results path |
| `--full` | off (half precision) | Use fp32 instead of half precision |
| `--gpu` | `None` (auto) | GPU index to use (`-1` = last GPU) |
| `--batch_size` | `16` | Batch size for NLL evals (hellaswag/mmlu/arc) |
| `--mmlu_n_shot` | `0` | Few-shot examples for MMLU |
| `--gsm8k_n_shot` | `8` | Few-shot examples for GSM8K |
| `--gsm8k_batch_size` | `4` | Batch size for GSM8K generation |
| `--humaneval_samples` | `1` | Completions per HumanEval problem (pass@k) |
| `--tok_kind` / `--tok_path` / `--special_tokens` | `None` (auto) | Tokenizer overrides; auto-detected from checkpoint if unset |
| `--shard_strategy` | `none` | Multi-GPU sharding: `auto`, `balanced`, or `none` |
| `--max_memory` | `None` | Max memory per GPU when sharding (e.g. `'14GiB'`) |
| `--qk_norm_mode` | `None` | QK norm override: `none`/`before_rope`/`after_rope_legacy`/`after_rope_fixed` |
| `--use_keel` | off | Enable KEEL (Highway-style Post-LN) |

## Notes
- Datasets are pulled at runtime via HuggingFace `datasets`: `cais/mmlu`,
  `allenai/ai2_arc`, `gsm8k`, `openai/openai_humaneval`. HellaSwag is read from a
  local `./hellaswag/` directory (via `hellaswag.hellaswag.iterate_examples`).
  Network/dataset-cache access is required; a failing test is caught, logged, and
  recorded with an `error` field rather than aborting the whole run.
- Relative paths assume the tool is run from the `tools/` directory: it inserts
  `../common_fsdp2` and `../saved_code` onto `sys.path`, logs to `./logs/eval_log.txt`,
  and resolves dataset/output paths relative to cwd.
- HumanEval executes model-generated code. Execution is sandboxed in a separate
  process (`multiprocessing.Process`) with a 10s timeout and several dangerous
  builtins/`os`/`shutil`/`subprocess` calls nulled out — but this is best-effort, not
  a true jail. It is Windows-compatible (no SIGALRM/resource limits). Treat generated
  code as untrusted.
- MMLU assumes exactly 4 choices per question; ARC handles a variable number of
  choices per example. Both score by argmin of answer-span loss.
- Output JSON contains `checkpoint`, `step`, `tokens`, `timestamp`, and a `results`
  map keyed by test name (each a serialized `TestResult`: `num_correct`, `num_total`,
  `accuracy` as a percentage, `duration_seconds`, optional `error`).
- Requires a CUDA device and the project conda env (`trainenv`); model loading goes
  through `nc.load_model_and_tokenizer`. Sibling tool: `generate_neo.py` (chat/generation
  and the shared eval primitives this script wraps).
