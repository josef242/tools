# chat_neo.py
> Interactive REPL chat/roleplay interface for testing model checkpoints — loads a custom `.pt` checkpoint (or a GGUF via llama_cpp), drives a streaming generation loop with slash-commands, and supports both raw-completion and chat-template (mara/llama/chatml) modes.

## What it does
This is a hands-on inference/chat harness for talking to a trained checkpoint at the terminal. You point it at a checkpoint file or run directory, it loads the model + tokenizer (via `neo_common.load_model_and_tokenizer` for custom checkpoints, or `llama_cpp.Llama` for `.gguf`), then drops you into an interactive prompt loop: pick a scenario YAML, then exchange turns with the model. Generation streams token-by-token through `neo_common.stream_generate_kv` (KV-cached). It is the manual/qualitative counterpart to batch eval scripts — reach for it to "feel out" a checkpoint, test persona/roleplay behavior, or sanity-check the mara chat template. Not part of the WD-waste / Newton-Schulz / z-loss probe families; it's a general-purpose interactive tool.

Two generation modes:
- **Raw completion** (default) — concatenates a `Name: text` transcript and completes it; stops on user-name patterns.
- **Chat completion** — `--chat_format mara` builds prompts with mara special tokens for custom checkpoints; any other `--chat_format` routes GGUF models through `create_chat_completion`.

## Usage
```bash
# Custom checkpoint, raw-completion roleplay (auto-selects newest model_*.pt in a run dir)
python chat_neo.py --model_path /path/to/run_dir

# Single checkpoint file, full precision, custom user name
python chat_neo.py --model_path /path/to/model_18500.pt --full --user "Alice"

# Mara chat-template mode (special tokens, matches training format)
python chat_neo.py --model_path /path/to/run_dir --chat_format mara

# GGUF model via llama_cpp with a chat format
python chat_neo.py --model_path /path/to/model.gguf --chat_format llama-3
```
On launch it loops asking `Enter prompt file [...]:` (a scenario YAML resolved against the hardcoded prompt dir `../xn/mpd/`), prints the loaded conversation, then enters the turn loop. Type to respond; `/help` lists commands.

## Key arguments
| flag | default | meaning |
|------|---------|---------|
| `--model_path` | (required) | Checkpoint `.pt`, `.gguf`, or a directory (auto-picks the highest-step `model_*.pt`). |
| `--temp` | `0.7` | Sampling temperature. |
| `--top_p` | `0.98` | Top-p (nucleus) sampling. |
| `--gen_size` | `128` | Max new tokens per generation. |
| `--context_len` | `4096` | Context length (custom models override this from `model_cfg.max_seq_len`). |
| `--full` | off | Use fp32 instead of half precision. |
| `--force` | off | Force-response mode (appends `{ai_name}:` to prompt before generating). |
| `--user` | `User` | User name(s), comma-separated; first is primary. Used for stop sequences in raw mode. |
| `--chat_format` | `None` | `mara` (custom special-token chat) or a GGUF chat format (`llama-3`, `chatml`, ...). Absent = raw completion. |
| `--use_keel` | off | Enable KEEL (Highway Post-LN) when the checkpoint was trained with it but its config omits the flag. |
| `--tok_kind` / `--tok_path` / `--special_tokens` | `None` | Tokenizer overrides; auto-detected from the checkpoint when omitted. |
| `--gpu` | `None` | GPU index. |
| `--shard_strategy` | `balanced` | Multi-GPU shard strategy (`auto`/`balanced`/`none`). |
| `--max_memory` | `None` | Max memory per GPU (custom models). |
| `--n_gpu_layers` | `-1` | GGUF: layers offloaded to GPU. |
| `--tensor_split` | `auto` | GGUF tensor split across GPUs (`auto` or comma list of fractions/percents). |
| `--think` | `compact` | Thinking-trace display for reasoning GGUF models (e.g. Qwen3.8): `full` streams the `<think>` block verbatim; `compact` shows a `[Thinking...N]` placeholder with a live in-place token count (`1.4k`-style above 1000). The trace is stripped from chat history in both modes. |

## In-session slash commands
`/` continue last message · `/rep` edit last message (multiline) · `/temp`, `/top`, `/gen` (token count) adjust sampling · `/force`, `/debug`, `/compact`, `/raw`, `/think` (thinking display full/compact) toggles · `/ls [-l] [glob]`, `/cd`, `/prompt`, `/name`, `/cls`, `/new`, `/exit`, `/help`. `//text` injects narrative/OOC text (a system message in chat mode, raw text in raw mode).

## Notes
- **Scenario input is YAML**, parsed by `ConversationConverter` (`char`/`ai_name`, `prompt`, `seed`, `conversations[]` with `role`/`content`). Supports `{{char}}`, `{{user}}`, `{{nl}}` template vars and inline `Name:`-prefixed transcript lines. Default scenario file `ev4.yaml` in dir `../xn/mpd/` — both are relative to CWD, so run from the expected working directory or use `/cd` and the prompt-file prompt.
- **PyTorch is lazy-loaded** — GGUF models skip importing torch entirely to save VRAM.
- **Dependencies:** custom checkpoints require `common_fsdp2/neo_common.py` (+ `tokenizer_abstraction`, and `../saved_code` on `sys.path` for FSDP1 checkpoints); GGUF requires `llama_cpp` (optional — a warning prints if missing). Optional `pynvml` for GPU auto-split; optional `common_fsdp2/logger` for logging to `./logs/chat_log.txt`.
- **Checkpoint selection:** a directory picks only `model_*.pt` by step number, deliberately ignoring `optim_*`, `rng_*`, `awd_*`, `ep_experts_*`, `moe_bias_*`.
- **Seed:** taken from the YAML `seed` (`-1` → random); seeds numpy and (if torch loaded) torch / CUDA.
- Comment header names the file `chat_neo_unified_v2.py`; this is the unified v2 chat interface.
- Output is unfiltered model generation in a roleplay setting — treat all generated/scenario text as opaque; do not copy it elsewhere.
