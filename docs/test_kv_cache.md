# test_kv_cache.py
> Tiny benchmark harness that runs `chat_neo.py` twice (KV cache off vs on) and reports the wall-clock speedup.

## What it does
This is a sanity/timing check for the KV cache implementation in `chat_neo.py`. It does **not** load a model itself — it shells out to `chat_neo.py` as a subprocess two times against the same checkpoint: once with `--no_kv_cache` and once in the default (cache-enabled) mode. It feeds a fixed short prompt on stdin, times each run, and prints a summary plus the percentage speedup if the cached run was faster. Reach for it after touching the KV cache path in `chat_neo.py` to confirm it still produces output and that caching is at least not slower. It is a standalone diagnostic, not part of any larger investigation family.

## Usage
There is **no argparse / CLI**. The script is run directly and prompts interactively for the checkpoint path on stdin:

```bash
python test_kv_cache.py
# then type the checkpoint path at the "Enter path to your model checkpoint:" prompt
```

It must be run from a directory where `chat_neo.py` is importable/launchable as `chat_neo.py` (it invokes `sys.executable chat_neo.py ...` with no path qualification, so run it from the same `tools/` dir as `chat_neo.py`).

## Notes
- **Fixed, hard-coded test parameters** (in `run_generation_test`): `--max_tokens 50`, `--temp 0.7`, and `--force` (force-response mode). The prompt sent on stdin is a fixed short string. None of these are configurable without editing the source.
- **Subprocess + timeout**: each run is launched via `subprocess.Popen`, fed the prompt through `stdin`, and capped at a 30-second timeout. A timeout or non-zero exit code is treated as a failed test (and a timeout is recorded as `30.0s` in the summary).
- **Speedup is wall-clock only**: it times whole-process latency (includes model load, not just generation), so the reported percentage is coarse. The script itself notes the speedup is most visible with longer generations / longer sequences, whereas this test only generates 50 tokens.
- **Order of runs**: it runs WITHOUT cache first, then WITH cache. Process/GPU warm-up is not controlled for, so a single run is not a rigorous benchmark.
- **Dependencies**: requires a working `chat_neo.py` (the actual inference/chat tool that loads the checkpoint) and whatever environment that tool needs (GPU, trainenv, model config alongside the checkpoint). See the `chat_neo.py` doc for checkpoint/path requirements.
