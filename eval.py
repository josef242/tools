# eval.py - Checkpoint Evaluation Tool
# Runs standard benchmarks on a model checkpoint:
# HellaSwag, MMLU, ARC-Easy, ARC-Challenge, GSM8K, HumanEval
#
# Usage:
#   python eval.py <checkpoint_path> --test hellaswag --test mmlu --test arc-easy
#   python eval.py <checkpoint_path> --test all
#   python eval.py <checkpoint_path> --test all --output my_results.json

import os
import re
import sys
import io
import time
import json
import argparse
import traceback
import multiprocessing
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict

import torch
from torch.nn import functional as F

# ------------------------- Common Files -------------------------
common_path = '../common_fsdp2'
if common_path not in sys.path:
    sys.path.insert(0, common_path)
saved_code_path = '../saved_code'
if saved_code_path not in sys.path:
    sys.path.insert(0, saved_code_path)

import logger
logger._instance.set_logdir("./logs")
logger._instance.set_default_logfile("eval_log.txt")
logger._instance.set_rank(0)

import neo_common as nc

# Import reusable functions from generate_neo
from generate_neo import (
    score_hellaswag_batch,
    pad_and_stack,
    get_batch_loss,
    resolve_model_path,
    get_checkpoint_info,
)

# Restore our logger settings (generate_neo import side-effect overwrites them)
logger._instance.set_default_logfile("eval_log.txt")
# ----------------------------------------------------------------

ALL_TESTS = ["hellaswag", "mmlu", "arc-easy", "arc-challenge", "gsm8k", "humaneval"]


@dataclass
class TestResult:
    test_name: str
    num_correct: int
    num_total: int
    accuracy: float  # percentage (0-100)
    duration_seconds: float
    error: Optional[str] = None


# =============================================================================
# HellaSwag wrapper
# =============================================================================

def run_hellaswag(model, tokenizer, device, pad_id, batch_size):
    """Run HellaSwag evaluation, returning (correct, total, accuracy)."""
    from hellaswag.hellaswag import iterate_examples

    all_examples = list(iterate_examples("val", data_dir="./hellaswag/"))
    num_correct = 0
    num_total = 0

    for i in range(0, len(all_examples), batch_size):
        batch = all_examples[i:i + batch_size]
        preds, gold = score_hellaswag_batch(model, tokenizer, batch, device, pad_id)
        preds = preds.cpu()
        gold = gold.cpu()
        torch.cuda.empty_cache()
        num_correct += (preds == gold).sum().item()
        num_total += gold.numel()
        acc = 100 * num_correct / num_total
        pct = 100 * (i + len(batch)) / len(all_examples)
        print(f"\r[HellaSwag] {i+len(batch)}/{len(all_examples)} [{pct:.1f}%] Acc: {acc:.2f}%", end="")

    print("")
    accuracy = 100.0 * num_correct / num_total
    logger.print_and_log(f"HellaSwag accuracy: {num_correct}/{num_total} = {accuracy:.2f}%")
    return num_correct, num_total, accuracy


# =============================================================================
# MMLU wrapper (duplicated loop from generate_neo.py to capture return values)
# =============================================================================

def run_mmlu(model, tokenizer, device, pad_id, batch_size, n_shot=0):
    """Run MMLU evaluation, returning (correct, total, accuracy)."""
    from datasets import load_dataset

    logger.print_and_log(f"Evaluating MMLU (cais/mmlu) with {n_shot}-shot prompting...")

    all_mmlu_subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging",
        "human_sexuality", "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
        "philosophy", "prehistory", "professional_accounting", "professional_law",
        "professional_medicine", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy", "virology",
        "world_religions"
    ]

    def format_mmlu_prompt(question, choices, answer=None):
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"
        if answer is not None:
            prompt += f" {chr(65+answer)}\n\n"
        return prompt

    def get_few_shot_examples(dataset_name, n_examples):
        if n_examples == 0:
            return ""
        dev_set = load_dataset("cais/mmlu", dataset_name, split="dev")
        few_shot_prompt = ""
        for i in range(min(n_examples, len(dev_set))):
            example = dev_set[i]
            few_shot_prompt += format_mmlu_prompt(
                example["question"], example["choices"], example["answer"]
            )
        return few_shot_prompt

    num_correct = 0
    num_total = 0

    for subject_name in all_mmlu_subjects:
        logger.print_and_log(f"Subject: {subject_name}")
        subset = load_dataset("cais/mmlu", subject_name, split="test")
        logger.print_and_log(f"Loaded {len(subset)} examples for {subject_name}")

        few_shot_prompt = get_few_shot_examples(subject_name, n_shot)
        subject_correct = 0
        subject_total = 0

        for batch_start in range(0, len(subset), batch_size):
            batch_end = min(batch_start + batch_size, len(subset))
            batch_examples = [subset[i] for i in range(batch_start, batch_end)]

            all_tokens = []
            all_masks = []
            labels = []

            for example in batch_examples:
                question = example["question"]
                choices = example["choices"]
                label = example["answer"]
                labels.append(label)

                test_prompt = format_mmlu_prompt(question, choices, answer=None)
                full_prompt = few_shot_prompt + test_prompt
                prompt_tokens = tokenizer.encode(full_prompt, bos=True, eos=False)
                prompt_length = len(prompt_tokens)

                for choice_idx in range(4):
                    answer_letter = chr(65 + choice_idx)
                    full_text = full_prompt + f" {answer_letter}"
                    tokens = torch.tensor(
                        tokenizer.encode(full_text, bos=True, eos=False)
                    ).unsqueeze(0)
                    mask = torch.zeros_like(tokens)
                    mask[:, prompt_length:] = 1
                    all_tokens.append(tokens)
                    all_masks.append(mask)

            batched_tokens = pad_and_stack(all_tokens, pad_id).to(device)
            batched_masks = pad_and_stack(all_masks, pad_id).to(device)
            losses = get_batch_loss(model, batched_tokens, batched_masks, device)
            losses = losses.view(len(batch_examples), 4)
            predictions = losses.argmin(dim=1)

            for pred, label in zip(predictions, labels):
                subject_total += 1
                if pred.item() == label:
                    subject_correct += 1

            if (batch_end % 100) <= batch_size:
                acc_so_far = (subject_correct / subject_total) * 100 if subject_total > 0 else 0
                print(f"\r[{subject_name}] {batch_end}/{len(subset)} Acc: {acc_so_far:.2f}%", end="")

        print("")

        if subject_total == 0:
            logger.print_and_log(f"No data for subject '{subject_name}' in cais/mmlu test set!")
            continue

        subject_acc = 100.0 * subject_correct / subject_total
        logger.print_and_log(f"Subject: {subject_name} Accuracy: {subject_correct}/{subject_total} = {subject_acc:.2f}%")

        num_correct += subject_correct
        num_total += subject_total

    if num_total == 0:
        logger.print_and_log("No valid data found for any subject.")
        return 0, 0, 0.0

    accuracy = 100.0 * num_correct / num_total
    logger.print_and_log(f"MMLU {n_shot}-shot overall accuracy: {num_correct}/{num_total} = {accuracy:.2f}%")
    return num_correct, num_total, accuracy


# =============================================================================
# GSM8K wrapper
# =============================================================================

def run_gsm8k(model, tokenizer, device, n_shot, batch_size):
    """Run GSM8K evaluation, returning (correct, total, accuracy)."""
    from generate_neo import test_gsm8k
    from datasets import load_dataset

    test_dataset = load_dataset("gsm8k", "main", split="test")
    num_total = len(test_dataset)

    accuracy = test_gsm8k(model, tokenizer, device, n_shot=n_shot, batch_size=batch_size)
    num_correct = int(round(accuracy * num_total / 100.0))
    return num_correct, num_total, accuracy


# =============================================================================
# ARC evaluation (new implementation)
# =============================================================================

def test_arc(model, tokenizer, device, pad_id, subset="ARC-Easy", batch_size=16):
    """
    Evaluate on ARC-Easy or ARC-Challenge using NLL-based scoring.
    Same approach as MMLU: encode question + each choice, compute NLL on answer token,
    pick the choice with the lowest loss.

    Args:
        subset: "ARC-Easy" or "ARC-Challenge"

    Returns:
        (num_correct, num_total, accuracy)
    """
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", subset, split="test")
    logger.print_and_log(f"Evaluating {subset} ({len(ds)} examples)...")

    num_correct = 0
    num_total = 0

    for batch_start in range(0, len(ds), batch_size):
        batch_end = min(batch_start + batch_size, len(ds))
        batch_examples = [ds[i] for i in range(batch_start, batch_end)]

        all_tokens = []
        all_masks = []
        gold_indices = []
        num_choices_per_example = []

        for example in batch_examples:
            question = example["question"]
            choices = example["choices"]["text"]
            choice_labels = example["choices"]["label"]  # e.g. ["A","B","C","D"] or ["1","2","3","4"]
            answer_key = example["answerKey"]

            # Find gold index
            gold_idx = choice_labels.index(answer_key)
            gold_indices.append(gold_idx)
            num_choices_per_example.append(len(choices))

            # Build prompt in the same format as MMLU
            prompt = f"Question: {question}\n"
            for label, choice in zip(choice_labels, choices):
                prompt += f"{label}. {choice}\n"
            prompt += "Answer:"

            prompt_tokens = tokenizer.encode(prompt, bos=True, eos=False)
            prompt_length = len(prompt_tokens)

            # Score each choice
            for label in choice_labels:
                full_text = prompt + f" {label}"
                tokens = torch.tensor(
                    tokenizer.encode(full_text, bos=True, eos=False)
                ).unsqueeze(0)
                mask = torch.zeros_like(tokens)
                mask[:, prompt_length:] = 1
                all_tokens.append(tokens)
                all_masks.append(mask)

        # Pad, stack, and compute losses
        batched_tokens = pad_and_stack(all_tokens, pad_id).to(device)
        batched_masks = pad_and_stack(all_masks, 0).to(device)
        losses = get_batch_loss(model, batched_tokens, batched_masks, device)

        # Map losses back to examples (variable number of choices per example)
        offset = 0
        for i, (n_choices, gold_idx) in enumerate(zip(num_choices_per_example, gold_indices)):
            example_losses = losses[offset:offset + n_choices]
            pred = example_losses.argmin().item()
            if pred == gold_idx:
                num_correct += 1
            num_total += 1
            offset += n_choices

        if (batch_end % 100) <= batch_size or batch_end == len(ds):
            acc = (num_correct / num_total) * 100 if num_total > 0 else 0
            pct = 100 * batch_end / len(ds)
            print(f"\r[{subset}] {batch_end}/{len(ds)} [{pct:.1f}%] Acc: {acc:.2f}%", end="")

    print("")
    accuracy = 100.0 * num_correct / num_total if num_total > 0 else 0.0
    logger.print_and_log(f"{subset} accuracy: {num_correct}/{num_total} = {accuracy:.2f}%")
    return num_correct, num_total, accuracy


# =============================================================================
# HumanEval - Windows-compatible sandboxed code execution
# =============================================================================

def _humaneval_worker(code_str, result_queue):
    """Execute code in a subprocess with basic safety guards."""
    import faulthandler
    faulthandler.disable()

    # Disable dangerous builtins
    import builtins
    builtins.exit = None
    builtins.quit = None

    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None

    import subprocess
    subprocess.Popen = None

    try:
        import contextlib
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        exec_globals = {}
        with contextlib.redirect_stdout(stdout_capture):
            with contextlib.redirect_stderr(stderr_capture):
                exec(code_str, exec_globals)
        result_queue.put({
            "success": True,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "error": None,
        })
    except Exception as e:
        result_queue.put({
            "success": False,
            "stdout": "",
            "stderr": "",
            "error": f"{type(e).__name__}: {e}",
        })


def execute_code_safe(code, timeout=10.0):
    """
    Execute Python code in a sandboxed subprocess.
    Windows-compatible (no SIGALRM, no resource limits).
    Uses a Queue instead of Manager to avoid spawning extra server processes.
    """
    result_queue = multiprocessing.Queue()

    p = multiprocessing.Process(target=_humaneval_worker, args=(code, result_queue))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.kill()
        p.join(timeout=2)
        return {"success": False, "error": "Timeout", "stdout": "", "stderr": ""}

    try:
        return result_queue.get_nowait()
    except Exception:
        return {"success": False, "error": "No result returned", "stdout": "", "stderr": ""}


def _extract_imports(prompt):
    """Extract import statements from the beginning of a code prompt."""
    imports = []
    for line in prompt.split('\n'):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            imports.append(stripped)
        elif stripped and not stripped.startswith('#'):
            break
    return '\n'.join(imports)


def _extract_code(completion):
    """Extract Python code from model completion, handling markdown code blocks."""
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, completion, re.DOTALL)
    if matches:
        return matches[0].strip()
    return completion.strip()


def test_humaneval(model, tokenizer, device, num_samples=1, temperature=0.0):
    """
    Evaluate on HumanEval (164 coding problems).
    Generates code completions and executes them against test suites.

    Args:
        num_samples: Number of completions per problem (for pass@k)
        temperature: 0.0 for greedy, >0 for sampling

    Returns:
        (num_correct, num_total, accuracy)
    """
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    num_total = len(ds)
    logger.print_and_log(f"Evaluating HumanEval ({num_total} problems, {num_samples} sample(s) each)...")

    num_correct = 0
    context_size = model.params.max_seq_len

    for i, problem in enumerate(ds):
        prompt = problem["prompt"]
        entry_point = problem["entry_point"]
        test_code = problem["test"]

        imports = _extract_imports(prompt)
        passed = False

        for sample_idx in range(num_samples):
            # Generate completion
            sample_temp = temperature if num_samples > 1 else 0.0
            sample_top_p = 0.95 if sample_temp > 0 else 1.0

            completion = nc.stream_generate_kv(
                model, tokenizer, prompt,
                max_new_tokens=512,
                context_size=context_size,
                temperature=sample_temp,
                top_p=sample_top_p,
                display=False,
                stop_on_eos=True,
            )

            # Build full program: imports + prompt + completion + tests + check
            code = _extract_code(completion)
            full_program = (
                imports + "\n\n"
                + prompt + code + "\n\n"
                + test_code + "\n"
                + f"check({entry_point})"
            )

            result = execute_code_safe(full_program, timeout=10.0)
            if result["success"]:
                passed = True
                break

        if passed:
            num_correct += 1

        if (i + 1) % 10 == 0 or (i + 1) == num_total:
            acc = 100.0 * num_correct / (i + 1)
            print(f"\r[HumanEval] {i+1}/{num_total} pass@{num_samples}: {acc:.1f}%", end="")

    print("")
    accuracy = 100.0 * num_correct / num_total if num_total > 0 else 0.0
    logger.print_and_log(f"HumanEval pass@{num_samples}: {num_correct}/{num_total} = {accuracy:.2f}%")
    return num_correct, num_total, accuracy


# =============================================================================
# Results summary and output
# =============================================================================

def print_summary(results, checkpoint_path, step, tokens):
    logger.print_and_log(f"\n{'='*65}")
    logger.print_and_log(f"EVALUATION SUMMARY")
    logger.print_and_log(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    if step is not None:
        token_str = f"{tokens / 1e6:.0f}M" if tokens else "unknown"
        logger.print_and_log(f"Step: {step}, Tokens: {token_str}")
    logger.print_and_log(f"{'='*65}")
    logger.print_and_log(f"{'Test':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>10} {'Time':>8}")
    logger.print_and_log(f"{'-'*65}")
    for r in results:
        if r.error:
            logger.print_and_log(f"{r.test_name:<20} {'ERROR':>10}   {r.error}")
        else:
            logger.print_and_log(
                f"{r.test_name:<20} {r.accuracy:>9.2f}% {r.num_correct:>10} {r.num_total:>10} {r.duration_seconds:>7.1f}s"
            )
    logger.print_and_log(f"{'='*65}")


def save_results(results, output_path, checkpoint_path, step, tokens):
    data = {
        "checkpoint": checkpoint_path,
        "step": step,
        "tokens": tokens,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {r.test_name: asdict(r) for r in results},
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.print_and_log(f"Results saved to: {output_path}")


# =============================================================================
# CLI argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Checkpoint evaluation tool - run standard benchmarks on a model checkpoint."
    )
    parser.add_argument("checkpoint", type=str,
                        help="Path to checkpoint .pt file or directory (auto-selects latest)")
    parser.add_argument("--test", action="append", dest="tests", default=None,
                        choices=ALL_TESTS + ["all"],
                        help="Test to run (can specify multiple). Use 'all' for all tests.")

    # Model loading args (matching generate_neo.py)
    parser.add_argument("--full", action="store_true",
                        help="Use full precision (fp32) instead of half precision")
    parser.add_argument("--tok_kind", type=str, default=None,
                        help="Tokenizer kind (auto-detected from checkpoint if not specified)")
    parser.add_argument("--tok_path", type=str, default=None,
                        help="Path to tokenizer files (auto-detected from checkpoint if not specified)")
    parser.add_argument("--special_tokens", type=str, default=None,
                        help="Path to special tokens JSON file (auto-detected from checkpoint if not specified)")
    parser.add_argument("--shard_strategy", type=str, default="none",
                        choices=["auto", "balanced", "none"],
                        help="Sharding strategy for multi-GPU")
    parser.add_argument("--max_memory", type=str, default=None,
                        help="Max memory per GPU when sharding (e.g., '14GiB')")
    parser.add_argument("--qk_norm_mode", type=str, default=None,
                        choices=[None, "none", "before_rope", "after_rope_legacy", "after_rope_fixed"],
                        help="QK normalization mode")
    parser.add_argument("--use_keel", action="store_true",
                        help="Enable KEEL (Highway-style Post-LN)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Which GPU to use (0, 1, etc; -1 for last GPU)")

    # Eval configuration
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for NLL-based evals: hellaswag, mmlu, arc (default: 16)")
    parser.add_argument("--mmlu_n_shot", type=int, default=0,
                        help="Number of few-shot examples for MMLU (default: 0)")
    parser.add_argument("--gsm8k_n_shot", type=int, default=8,
                        help="Number of few-shot examples for GSM8K (default: 8)")
    parser.add_argument("--gsm8k_batch_size", type=int, default=4,
                        help="Batch size for GSM8K generation (default: 4)")
    parser.add_argument("--humaneval_samples", type=int, default=1,
                        help="Number of samples per HumanEval problem (default: 1)")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output results file path (default: <model>_step<N>_results.json)")

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Resolve tests
    if args.tests is None:
        print("No tests specified. Use --test <name> (can repeat). Available tests:")
        for t in ALL_TESTS:
            print(f"  --test {t}")
        print(f"  --test all")
        sys.exit(1)

    if "all" in args.tests:
        tests_to_run = list(ALL_TESTS)
    else:
        tests_to_run = args.tests

    # Resolve checkpoint path
    checkpoint_path = resolve_model_path(args.checkpoint)
    step_number, token_count = get_checkpoint_info(checkpoint_path)

    # Determine output file
    if args.output:
        output_path = args.output
    else:
        # Build default name from model directory and step
        parent_dir = os.path.basename(os.path.dirname(os.path.abspath(checkpoint_path)))
        step_str = f"step{step_number}" if step_number else "unknown"
        output_path = f"{parent_dir}_{step_str}_results.json"

    logger.print_and_log(f"Checkpoint: {checkpoint_path}")
    if step_number is not None:
        token_str = f"{token_count / 1e6:.0f}M" if token_count else "unknown"
        logger.print_and_log(f"Step: {step_number}, Tokens: {token_str}")
    logger.print_and_log(f"Tests: {', '.join(tests_to_run)}")
    logger.print_and_log(f"Output: {output_path}")

    # Load model
    device = nc.detect_device(args.gpu)

    qk_mode = args.qk_norm_mode
    if qk_mode is not None:
        qk_mode = qk_mode.lower()
        if qk_mode == "none":
            qk_mode = None

    start_time = time.time()
    model, tokenizer, cfg = nc.load_model_and_tokenizer(
        checkpoint_path,
        device=device,
        half_precision=not args.full,
        tok_kind=args.tok_kind,
        tok_path=args.tok_path,
        special_tokens=args.special_tokens,
        shard_strategy=args.shard_strategy,
        preferred_gpu=args.gpu,
        max_memory_per_gpu=args.max_memory,
        qk_norm_mode=qk_mode,
        use_keel=args.use_keel or None,
    )
    logger.print_and_log(f"Model loaded in {time.time() - start_time:.2f} seconds")

    pad_id = tokenizer.pad_id

    # Run each test
    results = []
    for test_name in tests_to_run:
        logger.print_and_log(f"\n{'='*65}")
        logger.print_and_log(f"Running: {test_name}")
        logger.print_and_log(f"{'='*65}")
        start_time = time.time()

        try:
            if test_name == "hellaswag":
                nc_, nt_, acc = run_hellaswag(model, tokenizer, device, pad_id, args.batch_size)
            elif test_name == "mmlu":
                nc_, nt_, acc = run_mmlu(model, tokenizer, device, pad_id, args.batch_size, args.mmlu_n_shot)
            elif test_name == "arc-easy":
                nc_, nt_, acc = test_arc(model, tokenizer, device, pad_id, "ARC-Easy", args.batch_size)
            elif test_name == "arc-challenge":
                nc_, nt_, acc = test_arc(model, tokenizer, device, pad_id, "ARC-Challenge", args.batch_size)
            elif test_name == "gsm8k":
                nc_, nt_, acc = run_gsm8k(model, tokenizer, device, args.gsm8k_n_shot, args.gsm8k_batch_size)
            elif test_name == "humaneval":
                nc_, nt_, acc = test_humaneval(model, tokenizer, device, args.humaneval_samples)
            else:
                raise ValueError(f"Unknown test: {test_name}")

            duration = time.time() - start_time
            result = TestResult(test_name, nc_, nt_, acc, duration)

        except Exception as e:
            duration = time.time() - start_time
            logger.print_and_log(f"ERROR in {test_name}: {e}")
            traceback.print_exc()
            result = TestResult(test_name, 0, 0, 0.0, duration, error=str(e))

        results.append(result)
        logger.print_and_log(f"{test_name} completed in {result.duration_seconds:.1f}s")

    # Summary and output
    print_summary(results, checkpoint_path, step_number, token_count)
    save_results(results, output_path, checkpoint_path, step_number, token_count)


if __name__ == "__main__":
    main()
