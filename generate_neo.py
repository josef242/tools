# generate_neo.py
import os
import re
import sys
import torch
from torch.nn import functional as F
import time
from hellaswag.hellaswag import iterate_examples
import json
import yaml
import argparse
import time
from contextlib import nullcontext
import math
import numpy as np

# ------------------------- Common Files -------------------------
# Primary common path (FSDP2)
common_path = '../common_fsdp2'
if common_path not in sys.path:
    sys.path.insert(0, common_path)  # insert at the beginning to prioritize
# Also add saved_code for FSDP1 checkpoint support
saved_code_path = '../saved_code'
if saved_code_path not in sys.path:
    sys.path.insert(0, saved_code_path)
# Note: Transformer/ModelArgs imported dynamically in neo_common based on checkpoint version
import logger
logger._instance.set_logdir("./logs")
logger._instance.set_default_logfile("gen_log.txt")
logger._instance.set_rank(0)
from CommandFramework import CommandFramework
from tokenizer_abstraction import get_tokenizer, LlamaTokenizerAdapter
import neo_common as nc
import coherence_metrics as cm
from redact_coherence_log import redact_file as _redact_coherence_log
# ----------------------------------------------------------------

# -----------------------------------------------------------------------------
# Batched MMLU Evaluation with N-shot support
def test_mmlu(model, tokenizer, device, pad_id, batch_size=16, subjects=None, n_shot=0):
    """
    Evaluate the model on the MMLU dataset (cais/mmlu) with N-shot prompting.
    Now with proper prompt format and batching for significant speedup.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        device: Device to run on
        pad_id: Padding token ID
        batch_size: Batch size for inference
        subjects: List of subjects to evaluate (None = all)
        n_shot: Number of examples to include (0 for zero-shot, typically 5 for few-shot)
    """
    logger.print_and_log(f"Evaluating MMLU (cais/mmlu) with {n_shot}-shot prompting...")

    from datasets import load_dataset

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

    if subjects is None:
        subjects = all_mmlu_subjects
    if isinstance(subjects, str):
        subjects = [subjects]

    def format_mmlu_prompt(question, choices, answer=None):
        """Format a single MMLU example in the standard format."""
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"
        if answer is not None:
            # Include the answer for few-shot examples
            prompt += f" {chr(65+answer)}\n\n"
        return prompt

    def get_few_shot_examples(dataset_name, n_examples):
        """Get n_examples from the dev set for few-shot prompting."""
        if n_examples == 0:
            return ""
        
        dev_set = load_dataset("cais/mmlu", dataset_name, split="dev")
        few_shot_prompt = ""
        
        # Take first n_examples (you could also randomize)
        for i in range(min(n_examples, len(dev_set))):
            example = dev_set[i]
            few_shot_prompt += format_mmlu_prompt(
                example["question"], 
                example["choices"], 
                example["answer"]
            )
        
        return few_shot_prompt

    num_correct = 0
    num_total = 0

    for subject_name in subjects:
        logger.print_and_log(f"Subject: {subject_name}")

        subset = load_dataset("cais/mmlu", subject_name, split="test")
        logger.print_and_log(f"Loaded {len(subset)} examples for {subject_name}")

        # Get few-shot examples for this subject
        few_shot_prompt = get_few_shot_examples(subject_name, n_shot)
        few_shot_tokens = tokenizer.encode(few_shot_prompt, bos=True, eos=False) if few_shot_prompt else []

        subject_correct = 0
        subject_total = 0

        # Process in batches
        for batch_start in range(0, len(subset), batch_size):
            batch_end = min(batch_start + batch_size, len(subset))
            batch_examples = [subset[i] for i in range(batch_start, batch_end)]
            
            # Prepare batch data
            all_tokens = []
            all_masks = []
            labels = []
            
            for example in batch_examples:
                question = example["question"]
                choices = example["choices"]
                label = example["answer"]
                labels.append(label)
                
                # Format the test question (without answer)
                test_prompt = format_mmlu_prompt(question, choices, answer=None)
                
                # Combine few-shot examples with test question
                full_prompt = few_shot_prompt + test_prompt
                
                # Encode the full prompt
                prompt_tokens = tokenizer.encode(full_prompt, bos=True, eos=False)
                prompt_length = len(prompt_tokens)
                
                # Evaluate each choice
                for choice_idx in range(4):
                    # Add the answer letter (A, B, C, or D)
                    answer_letter = chr(65 + choice_idx)
                    full_text = full_prompt + f" {answer_letter}"
                    
                    tokens = torch.tensor(
                        tokenizer.encode(full_text, bos=True, eos=False)
                    ).unsqueeze(0)
                    
                    # Create mask - only compute loss on the answer token(s)
                    mask = torch.zeros_like(tokens)
                    mask[:, prompt_length:] = 1  # Only the answer part
                    
                    all_tokens.append(tokens)
                    all_masks.append(mask)
            
            # Pad and stack all sequences
            batched_tokens = pad_and_stack(all_tokens, pad_id).to(device)
            batched_masks = pad_and_stack(all_masks, pad_id).to(device)
            
            # Compute losses for the entire batch
            losses = get_batch_loss(model, batched_tokens, batched_masks, device)
            
            # Reshape losses to (num_examples, 4) and find best choice for each
            losses = losses.view(len(batch_examples), 4)
            predictions = losses.argmin(dim=1)
            
            # Update counts
            for pred, label in zip(predictions, labels):
                subject_total += 1
                if pred.item() == label:
                    subject_correct += 1
            
            # Progress update
            if (batch_end % 100) <= batch_size:
                acc_so_far = (subject_correct / subject_total) * 100 if subject_total > 0 else 0
                print(f"\r[{subject_name}] Example {batch_end}/{len(subset)} Accuracy: {acc_so_far:.2f}%", end="")

        print("")  # new line
        
        if subject_total == 0:
            logger.print_and_log(f"No data for subject '{subject_name}' in cais/mmlu test set!")
            continue

        subject_acc = 100.0 * subject_correct / subject_total
        logger.print_and_log(f"Subject: {subject_name} Accuracy: {subject_correct}/{subject_total} = {subject_acc:.2f}%")

        num_correct += subject_correct
        num_total += subject_total

    if num_total == 0:
        logger.print_and_log("No valid data found for any subject. Overall accuracy undefined.")
        print(f"--> MMLU {n_shot}-shot Overall Accuracy: N/A (No Data)")
        return

    overall_acc = 100.0 * num_correct / num_total
    logger.print_and_log(f"MMLU {n_shot}-shot overall accuracy: {num_correct}/{num_total} = {overall_acc:.2f}%")
    print(f"--> MMLU {n_shot}-shot Overall Accuracy: {overall_acc:.2f}%")


# -----------------------------------------------------------------------------
# helper functions for evaluations
def pad_and_stack(tensor_list, pad_value):
    max_len = max(t.size(1) for t in tensor_list)
    padded = [F.pad(t, (0, max_len - t.size(1)), value=pad_value)
              for t in tensor_list]
    return torch.cat(padded, dim=0)


def get_batch_loss(model, tokens, masks, device):
    with torch.no_grad():
        # Handle both string device and torch.device objects
        if isinstance(device, str):
            device_type = device
        else:
            device_type = device.type
            
        # Only use autocast for CUDA devices
        if device_type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(tokens)
                # Handle different output formats
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        else:
            # For CPU, MPS, or other devices, run without autocast
            outputs = model(tokens)
            # Handle different output formats
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()
        shift_masks = masks[..., 1:].contiguous()
        
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                               shift_tokens.view(-1), 
                               reduction='none')
        loss = loss.view(shift_tokens.size())
        masked_loss = (loss * shift_masks).sum(dim=1) / shift_masks.sum(dim=1).clamp(min=1)
    
    return masked_loss

def score_hellaswag_batch(model, tokenizer, examples, device, pad_id):
    """
    Vectorised, BOS-safe, length-aware HellaSwag scoring.
    """
    all_tok, all_mask, gold = [], [], []

    for ex in examples:            # one example = 4 endings
        ctx_ids = tokenizer.encode(ex['ctx'], bos=True, eos=False)
        gold.append(ex['label'])

        for ending in ex['endings']:
            # ① encode **only** the completion, but keep the leading space
            comp_ids = tokenizer.encode(" " + ending, bos=False, eos=False)

            ids     = torch.tensor(ctx_ids + comp_ids)
            mask    = torch.zeros_like(ids)
            mask[len(ctx_ids):] = 1         # ② mask only the completion

            all_tok.append(ids.unsqueeze(0))
            all_mask.append(mask.unsqueeze(0))

    # For sharded models (multi-GPU), get device from the embedding layer
    # This ensures input tensors are on the same device as tok_embeddings
    # Accelerate's dispatch_model expects inputs on the first module's device
    try:
        # Try to get the embedding layer device directly
        if hasattr(model, 'tok_embeddings'):
            input_device = model.tok_embeddings.weight.device
        # For HuggingFace models
        elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            input_device = model.model.embed_tokens.weight.device
        # Check if model has hf_device_map (Accelerate sharded model)
        elif hasattr(model, 'hf_device_map'):
            # Find the device of the first module in the device map
            first_device = list(model.hf_device_map.values())[0]
            input_device = torch.device(f"cuda:{first_device}" if isinstance(first_device, int) else first_device)
        else:
            input_device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        input_device = device

    tokens = pad_and_stack(all_tok,  pad_id).to(input_device)
    masks  = pad_and_stack(all_mask, 0).to(input_device)   # ③ pad mask with 0

    ll = get_batch_nll(model, tokens, masks)         # see below
    ll = ll.view(len(examples), 4)                   # (B, 4)

    # For sharded models, ll may be on a different device (output device)
    # Keep gold on the same device as ll for comparison
    return ll.argmin(dim=1), torch.tensor(gold, device=ll.device)


def get_batch_nll(model, tokens, masks):
    """Return **negative** log-likelihood (sum, not mean) per sequence."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(tokens)[0]                   # (B, T, V)
        # For sharded models, logits may be on a different device than tokens
        # Move tokens and masks to logits device for cross-entropy
        output_device = logits.device
        tokens_shifted = tokens[..., 1:].to(output_device)
        masks_shifted = masks[..., 1:].to(output_device)
        loss   = F.cross_entropy(
                    logits[..., :-1, :].reshape(-1, logits.size(-1)),
                    tokens_shifted.reshape(-1),
                    reduction='none').view(tokens.size(0), -1)
    return (loss * masks_shifted).sum(dim=1)       # ④ sum, no ÷len


# Batched HellaSwag evaluation
def test_hellaswag(model, tokenizer, device, pad_id, batch_size=16):
    """
    Evaluate on HellaSwag with proper batching.
    Handles tokenization boundaries correctly by ensuring space is part of completion.
    """
    num_correct = 0
    num_total   = 0
    all_examples = list(iterate_examples("val",data_dir="./hellaswag/"))
    #all_examples.sort(
    #    key=lambda ex: len(tokenizer.encode(ex["ctx"], bos=True, eos=False))
    #                   + max(len(ending) for ending in ex["endings"])
    #)

    t0 = time.time()
    for i in range(0, len(all_examples), batch_size):
        batch = all_examples[i:i + batch_size]
        preds, gold = score_hellaswag_batch(
            model, tokenizer, batch, device, pad_id
        )
        # ---- NEW: keep only CPU tensors; free GPU cache ----
        preds = preds.cpu()
        gold  = gold.cpu()
        torch.cuda.empty_cache()
        # -----------------------------------------------------
        num_correct += (preds == gold).sum().item()
        num_total   += gold.numel()

        done = i + len(batch)
        pct = 100 * done / len(all_examples)
        acc = 100 * num_correct / num_total
        elapsed = time.time() - t0
        if done < len(all_examples) and elapsed > 0:
            eta = elapsed * (len(all_examples) - done) / done
            eta_m, eta_s = divmod(int(eta), 60)
            eta_str = f"  ETA: {eta_m}m{eta_s:02d}s"
        else:
            eta_str = ""
        print(f"\rExample {done}/{len(all_examples)} "
              f"[{pct:5.1f}%]  Acc: {acc:5.2f}%{eta_str}   ", end="")    
            
    acc_norm = 100 * num_correct / num_total
    
    print("")
    logger.print_and_log(f"HellaSwag accuracy: {num_correct}/{num_total}={acc_norm:.2f}%")

    # Load and compare to the HellaSwag leaderboard
    with open("hellaswag/hella_leaderboard.json", "r") as f:
        leaderboard = json.load(f)
    
    last_beat = None
    to_beat = None
    for model_entry in reversed(leaderboard["models"]):
        if acc_norm > model_entry["score"]:
            last_beat = model_entry
        else:
            to_beat = model_entry
            break

    if to_beat is not None:
        logger.print_and_log(f"#{to_beat['rank']} - {to_beat['name']} with a score of {to_beat['score']}% in {to_beat['year']}")
    else:
        logger.print_and_log("Beat all models in the leaderboard!")
    logger.print_and_log(f"--> Your model HERE with a score of {acc_norm:.2f}%")
    if last_beat is not None:
        logger.print_and_log(f"#{last_beat['rank']} - {last_beat['name']} with a score of {last_beat['score']}% in {last_beat['year']}")
    else:
        logger.print_and_log("Did not beat any models in the leaderboard")

def test_perplexity_wikitext(model, tokenizer, device, version="wikitext-103-raw-v1", stride=512):
    """
    Evaluate perplexity on WikiText-103 or WikiText-2.
    This uses the sliding window approach that's standard in papers.
    """
    logger.print_and_log(f"Evaluating perplexity on {version}...")
    
    from datasets import load_dataset
    from tqdm import tqdm
    
    # Load the dataset
    if version == "wikitext-103-raw-v1":
        # dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", cache_dir="../datasets/eval/wikitext-103-raw-v1")
    else:  # wikitext-2-raw-v1
        # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir="../datasets/eval/wikitext-2-raw-v1")

    # Concatenate all text (standard practice)
    text = "\n\n".join([item['text'] for item in dataset if item['text'].strip()])
    
    # Tokenize the entire text
    encodings = tokenizer.encode(text, bos=True, eos=False)
    logger.print_and_log(f"Total tokens: {len(encodings):,}")
    
    # Sliding window evaluation (standard approach)
    max_length = model.params.max_seq_len
    total_loss = 0
    total_tokens = 0
    
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, len(encodings), stride)):
        end_loc = min(begin_loc + max_length, len(encodings))
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        
        input_ids = torch.tensor(encodings[begin_loc:end_loc], dtype=torch.long)
        input_ids = input_ids.unsqueeze(0).to(device)
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            # Handle device type for autocast
            device_type = device.type if hasattr(device, "type") else str(device).split(':')[0]
            
            if device_type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, _ = model(input_ids)
            else:
                logits, _ = model(input_ids)
            
            # Shift logits and labels for next-token prediction
            # We only compute loss on the tokens from prev_end_loc to end_loc
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            
            # Calculate which positions to include in loss
            # We want positions from (prev_end_loc - begin_loc) to (end_loc - begin_loc - 1)
            if begin_loc == 0:
                # First window: compute loss on all tokens except the first
                loss_mask = torch.ones_like(shift_labels, dtype=torch.bool)
            else:
                # Subsequent windows: only compute loss on new tokens
                loss_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                start_idx = prev_end_loc - begin_loc - 1  # -1 because of shifting
                loss_mask[..., start_idx:] = True
            
            # Compute loss only on masked positions
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )
            loss = loss.view(shift_labels.size())
            
            # Apply mask and sum
            masked_loss = loss * loss_mask.float()
            total_loss += masked_loss.sum().item()
            total_tokens += loss_mask.sum().item()
        
        prev_end_loc = end_loc
        if end_loc == len(encodings):
            break
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    logger.print_and_log(f"{version} Perplexity: {perplexity:.2f}")
    logger.print_and_log(f"Evaluated on {total_tokens:,} tokens")
    
    # Reference numbers for comparison
    if version == "wikitext-103-raw-v1":
        logger.print_and_log("Reference: GPT-2 (1.5B) = 17.48, GPT-3 (175B) ≈ 10.81")
    else:
        logger.print_and_log("Reference: GPT-2 (1.5B) = 29.41, GPT-3 (175B) ≈ 19.93")
    
    return perplexity

##############################################################################################################
# GSM8K Evaluation
##############################################################################################################
def test_gsm8k(model, tokenizer, device, n_shot=8, batch_size=4, num_samples=None, 
               majority_voting=1, temperature=0.0):
    """
    Evaluate on GSM8K (Grade School Math) problems.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        device: Device to run on
        n_shot: Number of examples in prompt (standard is 8)
        batch_size: Batch size for generation
        num_samples: Limit evaluation to this many examples (None = full test set)
        majority_voting: Number of samples to generate per question (>1 for voting)
        temperature: Temperature for sampling (0.0 for greedy, 0.7 for majority voting)
    """
    logger.print_and_log(f"Evaluating GSM8K ({n_shot}-shot, {majority_voting}x voting)...")
    
    from datasets import load_dataset
    import re
    from collections import Counter
    from tqdm import tqdm
    
    # Load dataset
    train_dataset = load_dataset("gsm8k", "main", split="train")
    test_dataset = load_dataset("gsm8k", "main", split="test")
    
    if num_samples:
        test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))
    
    # Get few-shot examples (use first n_shot from training set, or specific ones)
    # These are the examples used in the original paper for 8-shot
    standard_indices = [0, 4, 8, 12, 16, 20, 24, 28]  # Commonly used indices
    
    few_shot_examples = []
    indices_to_use = standard_indices[:n_shot] if n_shot <= 8 else range(n_shot)
    
    for idx in indices_to_use:
        ex = train_dataset[idx]
        few_shot_examples.append((ex['question'], ex['answer']))
    
    # Build few-shot prompt
    def build_prompt(question, few_shot_examples):
        prompt = ""
        for q, a in few_shot_examples:
            # Format each example
            prompt += f"Q: {q}\n"
            prompt += f"A: {a}\n\n"
        
        # Add the test question
        prompt += f"Q: {question}\n"
        prompt += "A:"
        return prompt
    
    def extract_answer(text):
        """Extract numerical answer from model output."""
        # First, look for #### pattern (most reliable)
        if '####' in text:
            after_hash = text.split('####')[-1].strip()
            # Get the first number after ####
            numbers = re.findall(r'-?\d+\.?\d*', after_hash)
            if numbers:
                return numbers[0].replace(',', '')
        
        # Fallback: Look for "answer is X" or "= X" patterns at the end
        patterns = [
            r'answer is[:\s]+(-?\d+\.?\d*)',
            r'=\s*(-?\d+\.?\d*)\s*(?:\.|$)',
            r'total of\s+(-?\d+\.?\d*)',
            r'therefore,?\s+(-?\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return matches[-1].replace(',', '')
        
        # Last resort: final number in the text
        all_numbers = re.findall(r'-?\d+\.?\d*', text)
        if all_numbers:
            return all_numbers[-1].replace(',', '')
        
        return ""
    
    def generate_batch(prompts, temperature):
        """Generate responses for a batch of prompts."""
        # Tokenize all prompts
        all_input_ids = []
        max_prompt_len = 0
        
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, bos=True, eos=False)
            all_input_ids.append(input_ids)
            max_prompt_len = max(max_prompt_len, len(input_ids))
        
        # Pad to same length
        padded_inputs = []
        attention_masks = []
        
        for input_ids in all_input_ids:
            padding_len = max_prompt_len - len(input_ids)
            padded = [tokenizer.pad_id] * padding_len + input_ids  # Left padding for generation
            mask = [0] * padding_len + [1] * len(input_ids)
            
            padded_inputs.append(padded)
            attention_masks.append(mask)
        
        batch_input = torch.tensor(padded_inputs, dtype=torch.long).to(device)
        batch_mask = torch.tensor(attention_masks, dtype=torch.long).to(device)
        
        # Generate
        with torch.no_grad():
            # You'll need to adapt this to your generation function
            # This is a simplified version - use your actual generation code
            responses = []
            for i in range(len(prompts)):
                # Use your nc.stream_generate or equivalent
                response = nc.stream_generate_kv(
                    model, tokenizer, prompts[i],
                    max_new_tokens=256,
                    context_size=model.params.max_seq_len,
                    temperature=temperature,
                    top_p=0.95 if temperature > 0 else 1.0,
                    display=False
                )
                responses.append(response)
        
        return responses
    
    # Evaluate
    correct = 0
    total = 0
    
    # Process in actual batches
    #for batch_start in range(0, len(test_dataset), batch_size):
    for batch_start in tqdm(range(0, len(test_dataset), batch_size), 
                       desc="Evaluating GSM8K", 
                       total=len(test_dataset)//batch_size + 1):
        batch_end = min(batch_start + batch_size, len(test_dataset))
        batch_questions = []
        batch_gold_answers = []
        
        # Prepare batch
        for idx in range(batch_start, batch_end):
            example = test_dataset[idx]
            batch_questions.append(example['question'])
            # Extract gold answer
            gold = example['answer'].split('####')[-1].strip().replace(',', '')
            batch_gold_answers.append(gold)
        
        # Generate responses (with multiple samples if using majority voting)
        all_predictions = []
        
        for sample_idx in range(majority_voting):
            prompts = [build_prompt(q, few_shot_examples) for q in batch_questions]
            responses = generate_batch(prompts, temperature)
            
            # Extract answers from responses
            predictions = [extract_answer(resp) for resp in responses]
            all_predictions.append(predictions)
        
        # Aggregate predictions (majority voting if multiple samples)
        for i, gold_answer in enumerate(batch_gold_answers):
            if majority_voting == 1:
                pred_answer = all_predictions[0][i]
            else:
                # Majority voting
                candidates = [all_predictions[j][i] for j in range(majority_voting)]
                # Filter out empty predictions
                candidates = [c for c in candidates if c]
                if candidates:
                    # Most common answer wins
                    counter = Counter(candidates)
                    pred_answer = counter.most_common(1)[0][0]
                else:
                    pred_answer = ""
            
            # Compare answers (handle different number formats)
            def normalize_number(s):
                """Normalize number for comparison."""
                s = s.replace(',', '').strip()
                try:
                    # Convert to float then back to standardize
                    num = float(s)
                    # If it's a whole number, return as int
                    if num == int(num):
                        return str(int(num))
                    else:
                        return f"{num:.6f}".rstrip('0').rstrip('.')
                except:
                    return s
            
            norm_pred = normalize_number(pred_answer)
            norm_gold = normalize_number(gold_answer)
            
            if norm_pred == norm_gold:
                correct += 1
            total += 1
            
            # Progress update
            if total % 10 == 0:
                acc = 100 * correct / total
                print(f"\rProgress: {total}/{len(test_dataset)} | Accuracy: {acc:.1f}%", end="")
    
    print()  # New line
    accuracy = 100 * correct / total
    logger.print_and_log(f"GSM8K Accuracy: {correct}/{total} = {accuracy:.2f}%")
    
    # Reference scores
    references = {
        "GPT-3 175B (8-shot)": 57.1,
        "LLaMA-2-7B (8-shot)": 14.6,
        "LLaMA-2-70B (8-shot)": 56.8,
        "GPT-4 (8-shot)": 92.0,
    }
    logger.print_and_log("Reference scores:")
    for model_name, score in references.items():
        logger.print_and_log(f"  {model_name}: {score}%")
    
    return accuracy

# -----------------------------------------------------------------------------
# stream_generate_kv has been moved to neo_common.py with ESC support
# -----------------------------------------------------------------------------


## Usage examples:
## Standard evaluation (8-shot, greedy)
#accuracy = test_gsm8k(model, tokenizer, device)
#
## With majority voting (40 samples, like in some papers)
#accuracy = test_gsm8k(model, tokenizer, device, 
#                      majority_voting=40, 
#                      temperature=0.7)
#
## Quick test with fewer examples
#accuracy = test_gsm8k(model, tokenizer, device, 
#                      num_samples=100)

def resolve_model_path(model_path):
    """
    Resolve model path to a specific .pt file.

    If model_path points to a directory, finds the most recent .pt file.
    If model_path points to a file, returns it as-is.

    Args:
        model_path: Path to model file or directory

    Returns:
        str: Path to the resolved .pt file

    Raises:
        FileNotFoundError: If no valid .pt file is found
    """
    # If it's a file, return it directly
    if os.path.isfile(model_path):
        return model_path

    # If it's a directory, find the most recent model checkpoint .pt file
    if os.path.isdir(model_path):
        pt_files = []
        for file in os.listdir(model_path):
            # Only match model checkpoint files, not auxiliary files like
            # ep_experts_step_*.pt, moe_bias_step_*.pt, optim_*.pt, rng_*.pt, awd_*.pt
            if file.startswith("model_") and file.endswith('.pt'):
                full_path = os.path.join(model_path, file)
                step_match = re.search(r'_(\d+)\.pt', file)
                if step_match:
                    step_number = int(step_match.group(1))
                    pt_files.append((step_number, full_path))

        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in directory: {model_path}")

        # Sort by step number and get the highest one
        pt_files.sort(reverse=True)
        selected_step, selected_path = pt_files[0]

        logger.print_and_log(f"Auto-selected checkpoint: {os.path.basename(selected_path)} (step {selected_step})")
        return selected_path

    # Path doesn't exist
    raise FileNotFoundError(f"Model path not found: {model_path}")

def get_checkpoint_info(checkpoint_path):
    """
    Extract step number and total token count for a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file (e.g., "./models/model_500.pt")

    Returns:
        tuple: (step_number, token_count) or (None, None) if not found
    """

    # Extract step number from checkpoint filename
    basename = os.path.basename(checkpoint_path)
    step_match = re.search(r'_(\d+)\.pt', basename)

    if not step_match:
        print(f"Warning: Could not extract step number from checkpoint name: {basename}")
        return None, None

    step_number = int(step_match.group(1))

    # Look for val_log.txt in the checkpoint directory
    checkpoint_dir = os.path.dirname(checkpoint_path)
    val_log_path = os.path.join(checkpoint_dir, "val_log.txt")

    if not os.path.exists(val_log_path):
        print(f"Warning: val_log.txt not found in {checkpoint_dir}")
        return step_number, None

    # Parse val_log.txt to find the matching step
    token_count = None

    try:
        with open(val_log_path, 'r') as f:
            for line in f:
                # Parse the line to extract step and token count
                # Format: date | time | st: STEP | tok: TOKEN_COUNT | ...

                # Look for "st:" followed by the step number
                st_match = re.search(r'st:\s*(\d+)', line)
                if st_match and int(st_match.group(1)) == step_number:
                    # Found the right line, extract token count
                    tok_match = re.search(r'tok:\s*(\d+)', line)
                    if tok_match:
                        token_count = int(tok_match.group(1))
                        break

        if token_count is None:
            print(f"Warning: Step {step_number} not found in val_log.txt")

    except Exception as e:
        print(f"Error reading val_log.txt: {e}")

    return step_number, token_count

class Generate(CommandFramework):
    def __init__(self, prefix, preferred_gpu=None):
        super().__init__(prefix)

        # default values
        self.gen_size = 512
        self.prompt_file_dir = "../xn/gen/"
        self.prompt_file = "ev4.yaml"
        self.temp = 0.7
        self.top_p = 0.9
        self.half = False
        self.model = None
        self.enc = None
        self.cfg = None
        self.device = nc.detect_device(preferred_gpu)
        self.eval_batch_size = 16  # Default batch size for evaluations
        self.user = ["User"]  # Default user names

        # new attributes for model loading (None = auto-detect from checkpoint)
        self.tok_kind = None
        self.tok_path = None
        self.special_tokens = None  # path to special tokens JSON file
        self.shard_strategy = "none"  # default shard strategy
        self.max_memory = None  # default max memory
        self.qk_norm_mode = None  # default qk norm mode (None, "before_rope", "after_rope_legacy", "after_rope_fixed")
        self.use_keel = False  # KEEL (Highway-style Post-LN)
        self.preferred_gpu = preferred_gpu  # store the preferred GPU
        self.model_path = None  # will be set when model is loaded

        self.add_command("temp", self.cmd_temp, "Change the temperature")
        self.add_command("top_p", self.cmd_top_p, "Change the top_p")
        self.add_command("size", self.cmd_size, "Change the generation size")
        self.add_command("prompt", self.cmd_prompt, "Load prompt file")
        self.add_command("chat", self.cmd_chat, "Load prompt file with chat format")
        self.add_command("dprompt", self.cmd_dprompt, "Manually enter a prompt")
        self.add_command("hella", self.cmd_hella, "Run HellaSwag evaluation")
        self.add_command("hella_sweep", self.cmd_hella_sweep, "Run HellaSwag sweep across training checkpoints")
        self.add_command("coherence_sweep", self.cmd_coherence_sweep, "Run coherence-metric sweep across training checkpoints")
        self.add_command("export", self.cmd_export, "Export the model to a .bin file")
        self.add_command("exit", self.cmd_exit, "Exit the program")
        self.add_command("ls", self.cmd_ls, "List all prompt files")
        self.add_command("cd", self.cmd_cd, "Change prompt file directory")
        self.add_command("mmlu", self.cmd_mmlu, "Run MMLU evaluation")
        self.add_command("batch", self.cmd_batch, "Change evaluation batch size")
        self.add_command("ppl", self.cmd_ppl, "Run perplexity evaluation on WikiText datasets")
        self.add_command("gsm8k", self.cmd_gsm8k, "Run GSM8K math evaluation")
        self.add_command("user", self.cmd_user, "Change the user names for prompts")
        self.add_command("load", self.cmd_load, "Load a new model checkpoint")
        self.add_command("cls", self.cmd_cls, "Clear the screen")

    def cmd_cls(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        return ""

    def cmd_load(self):
        """Load a new model checkpoint without exiting."""
        # Get the new model path
        model_path = input(f"Model path: [{self.model_path}] ")
        if not model_path:
            model_path = self.model_path

        # Resolve model path (handles both files and directories)
        try:
            model_path = resolve_model_path(model_path)
        except FileNotFoundError as e:
            return f"Error: {str(e)}"
        
        # Get tokenizer settings
        tok_kind = input(f"Tokenizer type (llama/hf): [{getattr(self, 'tok_kind', 'llama')}] ")
        if not tok_kind:
            tok_kind = getattr(self, 'tok_kind', 'llama')
        
        tok_path = input(f"Tokenizer path: [{getattr(self, 'tok_path', '../superbpe/superbpe+')}] ")
        if not tok_path:
            tok_path = getattr(self, 'tok_path', '../superbpe/superbpe+')
        
        # Ask about other settings
        change_settings = input("Change advanced settings? (y/N): ").lower() == 'y'
        
        if change_settings:
            # Half precision
            half_str = input(f"Use half precision? (y/n): [{'y' if self.half else 'n'}] ")
            if half_str:
                half = half_str.lower() == 'y'
            else:
                half = self.half
            
            # Sharding strategy
            shard_strategy = input(f"Shard strategy (auto/balanced/none): [{getattr(self, 'shard_strategy', 'none')}] ")
            if not shard_strategy:
                shard_strategy = getattr(self, 'shard_strategy', 'none')
            
            # Max memory
            max_memory = input(f"Max memory per GPU (e.g., 14GiB): [{getattr(self, 'max_memory', 'None')}] ")
            if not max_memory or max_memory.lower() == 'none':
                max_memory = None

            # QK norm mode
            print("QK norm modes: none, before_rope, after_rope_legacy, after_rope_fixed")
            qk_norm_input = input(f"QK norm mode: [{self.qk_norm_mode or 'none'}] ").strip().lower()
            if qk_norm_input in ("", "none"):
                qk_norm_mode = None
            elif qk_norm_input in ("before_rope", "after_rope_legacy", "after_rope_fixed"):
                qk_norm_mode = qk_norm_input
            else:
                print(f"Invalid mode '{qk_norm_input}', using None")
                qk_norm_mode = None
                
        else:
            # Use existing settings
            half = self.half
            shard_strategy = getattr(self, 'shard_strategy', 'none')
            max_memory = getattr(self, 'max_memory', None)
            qk_norm_mode = self.qk_norm_mode
        
        print(f"\nLoading new model: {model_path}")
        print(f"Tokenizer: {tok_kind} at {tok_path}")
        print(f"Settings: half={half}, shard={shard_strategy}, qk_norm_mode={qk_norm_mode}")

        try:
            # Clean up old model first
            if hasattr(self, 'model') and self.model is not None:
                print("Cleaning up old model...")
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Load the new model
            start_time = time.time()
            self.model, self.enc, self.cfg = nc.load_model_and_tokenizer(
                model_path,
                device=self.device,
                half_precision=half,
                tok_kind=tok_kind,
                tok_path=tok_path,
                special_tokens=self.special_tokens,
                shard_strategy=shard_strategy,
                preferred_gpu=getattr(self, 'preferred_gpu', None),
                max_memory_per_gpu=max_memory,
                qk_norm_mode=qk_norm_mode,
                use_keel=self.use_keel or None
            )

            # Test the new helper function
            step_number, token_count = get_checkpoint_info(model_path)
            logger.print_and_log(f"Step number: {step_number}, Total tokens: {token_count}")
            
            # Update stored settings
            self.model_path = model_path
            self.half = half
            self.tok_kind = tok_kind
            self.tok_path = tok_path
            self.shard_strategy = shard_strategy
            self.max_memory = max_memory
            self.qk_norm_mode = qk_norm_mode

            load_time = time.time() - start_time
            logger.print_and_log(f"Model loaded successfully in {load_time:.2f} seconds")

            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.print_and_log(f"Model has {total_params/1e9:.2f}B parameters")

            # model_path is guaranteed to be a string here (we validated it earlier)
            return f"Model '{os.path.basename(model_path)}' loaded successfully!"
            
        except Exception as e:
            logger.print_and_log(f"Error loading model: {str(e)}")
            return f"Failed to load model: {str(e)}"

    def cmd_exit(self):
        print("Goodbye!")
        exit(0)

    def cmd_user(self):
        val = input(f"New user names (comma-separated): [{', '.join(self.user)}] ")
        if val:
            self.user = [name.strip() for name in val.split(",") if name.strip()]
        
        return f"User names set to: {', '.join(self.user)}"

    def cmd_temp(self):
        val = input(f"New temperature: [{self.temp}] ")
        if val:
            self.temp = float(val)
        
        return f"Temperature set to {self.temp}"

    def cmd_top_p(self):
        val = input(f"New top_p: [{self.top_p}] ")
        if val:
            self.top_p = float(val)

        return f"Top_p set to {self.top_p}"

    def cmd_size(self):
        val = input(f"New output size: [{self.gen_size}] ")
        if val:
            self.gen_size = int(val)
        return f"Generation size set to {self.gen_size}"
    
    def cmd_batch(self):
        val = input(f"New evaluation batch size: [{self.eval_batch_size}] ")
        if val:
            self.eval_batch_size = int(val)
        return f"Evaluation batch size set to {self.eval_batch_size}. Larger values use more memory but run faster."

    def cmd_prompt(self):
        cur_prompt_file = self.prompt_file_dir + self.prompt_file
        val = input(f"New prompt file: [{cur_prompt_file}] ")
        if val:
            self.prompt_file = val
            cur_prompt_file = self.prompt_file_dir + self.prompt_file

        prompt, assistant, seed = nc.load_prompt(cur_prompt_file, self.user)
        if prompt is None:
            print(f"Error loading prompt file: {cur_prompt_file}")
            return f"<END>"
        
        if (seed==-1):
            seed = np.random.randint(0, 2**32 - 1, dtype=np.int64)

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            
        print(f"Seed: {seed}")

        # Calculate tokens per second
        start_time = time.time()
        nc.stream_generate_kv(
            self.model, self.enc, prompt,
            self.gen_size, self.cfg.max_seq_len,
            self.temp, self.top_p, display=True
        )
        print("")  # Newline after streaming
        end_time = time.time()
        elapsed = end_time - start_time
        tps =  self.gen_size/ elapsed if elapsed > 0 else 0.0
        print(f"Generated {self.gen_size} tokens in {elapsed:.2f} seconds ({tps:.2f} tokens/sec)")
        return None  # Text already displayed via streaming

    def cmd_chat(self):
        """Load a YAML prompt and generate using chat format with special tokens."""
        # Verify special tokens are registered in tokenizer
        required_tokens = ["<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>"]
        missing_tokens = []
        for token in required_tokens:
            test_ids = self.enc.encode(token, bos=False, eos=False)
            # If it encodes to more than 1 token, it's not registered as a special token
            if len(test_ids) != 1:
                missing_tokens.append(token)

        if missing_tokens:
            print(f"WARNING: Special tokens not registered in tokenizer: {missing_tokens}")
            print("Chat format requires --special_tokens parameter or checkpoint with special_tokens metadata.")
            print("Without proper special tokens, the model won't understand the chat structure.")
            confirm = input("Continue anyway? [y/N] ")
            if confirm.lower() != 'y':
                return "<END>"

        cur_prompt_file = self.prompt_file_dir + self.prompt_file
        val = input(f"Chat prompt file: [{cur_prompt_file}] ")
        if val:
            self.prompt_file = val
            cur_prompt_file = self.prompt_file_dir + self.prompt_file

        system_prompt, conversations, ai_name, seed = nc.load_yaml_chat_prompt(cur_prompt_file, self.user)
        if system_prompt is None:
            print(f"Error loading chat prompt file: {cur_prompt_file}")
            return "<END>"

        if seed == -1:
            seed = np.random.randint(0, 2**32 - 1, dtype=np.int64)

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        print(f"Seed: {seed}")

        # Render conversation in chat format
        chat_prompt = nc.render_chat_for_completion(system_prompt, conversations)

        # Role names for pretty printing
        user_name = self.user[0] if self.user else "User"
        role_names = {"assistant": ai_name, "user": user_name}

        # Pretty print mode is enabled by default for chat
        pretty_print = True
        print("-" * 50)

        # Generate with pretty print enabled
        start_time = time.time()
        nc.stream_generate_kv(
            self.model, self.enc, chat_prompt,
            self.gen_size, self.cfg.max_seq_len,
            self.temp, self.top_p, display=True,
            pretty_print=pretty_print, role_names=role_names
        )
        print("")  # Newline after streaming
        end_time = time.time()
        elapsed = end_time - start_time
        tps = self.gen_size / elapsed if elapsed > 0 else 0.0
        print("-" * 50)
        print(f"Generated {self.gen_size} tokens in {elapsed:.2f} seconds ({tps:.2f} tokens/sec)")
        return None  # Text already displayed via streaming

    def cmd_dprompt(self):
        prompt = input("Enter a prompt: ")
        prompt = prompt.strip().replace("\n", "")

        nc.stream_generate_kv(
            self.model, self.enc, prompt,
            self.gen_size, self.cfg.max_seq_len,
            self.temp, self.top_p, display=True
        )
        print("")  # Newline after streaming
        return None  # Text already displayed via streaming

    def cmd_ls(self) -> str:
        print(f"Files in {self.prompt_file_dir}:")
        for file in os.listdir(self.prompt_file_dir):
            print(f"   {file}")
        return ""

    def cmd_cd(self) -> str:
        val = input(f"Prompt directory [{self.prompt_file_dir}]: ")
        if val:
            # Normalize path
            val = val.replace("\\", "/")
            if not val.endswith("/"):
                val += "/"
            if os.path.isdir(val):
                self.prompt_file_dir = val
                print(f"Prompt directory: {self.prompt_file_dir}")
            else:
                print(f"Directory not found: {val}")
        else:
            print(f"Prompt directory: {self.prompt_file_dir}")
        return ""
    
    def cmd_gsm8k(self):
        num_samples = input("Number of samples to test (enter for all): ")
        num_samples = int(num_samples) if num_samples else None
        test_gsm8k(self.model, self.enc, self.device, 
                batch_size=self.eval_batch_size, num_samples=num_samples)
        
        # return f"GSM8K evaluation completed with {num_samples if num_samples else 'all'} samples."
        return ""
    
    def cmd_ppl(self):
        print("Select dataset:")
        print("1. WikiText-2 (quick, ~2MB)")
        print("2. WikiText-103 (standard, ~500MB)")
        print("3. Custom text file")
        
        choice = input("Choice [1]: ") or "1"
        
        if choice == "1":
            test_perplexity_wikitext(self.model, self.enc, self.device, "wikitext-2-raw-v1", stride=512)
            logger.print_and_log("Check here for score comparison: https://paperswithcode.com/sota/language-modeling-on-wikitext-2-raw-v1")
        elif choice == "2":
            test_perplexity_wikitext(self.model, self.enc, self.device, "wikitext-103-raw-v1", stride=1024)
            logger.print_and_log("Check here for score comparison: https://paperswithcode.com/sota/language-modeling-on-wikitext-103-raw-v1")
            #Excellent: < 6.0
            #Good: 6.0-8.0
            #Decent: 8.0-10.0
            logger.print_and_log("Rough benchmarks for WikiText-103:")
            logger.print_and_log("  Excellent: < 6.0")
            logger.print_and_log("  Good: 6.0-8.0")
            logger.print_and_log("  Decent: 8.0-10.0")
        elif choice == "3":
            file_path = input("Path to text file: ")
            # Custom implementation here - TODO
            print("Custom text file evaluation not implemented yet.")

        return ""
    
    def cmd_hella(self):
        test_hellaswag(self.model, self.enc, self.device, self.enc.pad_id, batch_size=self.eval_batch_size)

    def cmd_hella_sweep(self):
        """
        Interactive wrapper for run_hella_sweep.
        Prompts for parameters then calls the shared implementation.
        """
        # Get the log directory
        log_dir = input("Log directory path: ")
        if not log_dir or not os.path.isdir(log_dir):
            return f"Error: Invalid directory: {log_dir}"

        # Get optional parameters
        token_interval_str = input("Token interval in millions [500]: ") or "500"
        token_interval = int(token_interval_str) * 1_000_000

        # Call the shared implementation (save_step auto-detected)
        self.run_hella_sweep(log_dir, token_interval, interactive=True)
        return ""

    def _run_hellaswag_for_sweep(self):
        """
        Run HellaSwag evaluation and return (num_correct, num_total) instead of just printing.
        """
        num_correct = 0
        num_total = 0
        all_examples = list(iterate_examples("val", data_dir="./hellaswag/"))

        t0 = time.time()
        for i in range(0, len(all_examples), self.eval_batch_size):
            batch = all_examples[i:i + self.eval_batch_size]
            preds, gold = score_hellaswag_batch(
                self.model, self.enc, batch, self.device, self.enc.pad_id
            )
            preds = preds.cpu()
            gold = gold.cpu()
            torch.cuda.empty_cache()

            num_correct += (preds == gold).sum().item()
            num_total += gold.numel()

            done = i + len(batch)
            pct = 100 * done / len(all_examples)
            acc = 100 * num_correct / num_total
            elapsed = time.time() - t0
            if done < len(all_examples) and elapsed > 0:
                eta = elapsed * (len(all_examples) - done) / done
                eta_m, eta_s = divmod(int(eta), 60)
                eta_str = f"  ETA: {eta_m}m{eta_s:02d}s"
            else:
                eta_str = ""
            print(f"\rExample {done}/{len(all_examples)} [{pct:5.1f}%] Acc: {acc:5.2f}%{eta_str}   ", end="")

        print("")  # newline
        return num_correct, num_total

    def _detect_save_step(self, log_dir):
        """
        Auto-detect the checkpoint save interval by examining model files.

        Returns:
            int: Detected save step, or None if unable to detect
        """
        # Find all model checkpoint files
        checkpoint_files = []
        for f in os.listdir(log_dir):
            match = re.match(r'model_step_(\d+)\.pt', f)
            if match:
                checkpoint_files.append(int(match.group(1)))

        if len(checkpoint_files) < 2:
            return None

        # Sort and compute deltas between consecutive checkpoints
        checkpoint_files.sort()
        deltas = [checkpoint_files[i+1] - checkpoint_files[i]
                  for i in range(len(checkpoint_files) - 1)]

        # The save step is the most common delta (mode)
        from collections import Counter
        delta_counts = Counter(deltas)
        most_common_delta = delta_counts.most_common(1)[0][0]

        return most_common_delta

    def run_hella_sweep(self, log_dir, token_interval, interactive=False):
        """
        Run HellaSwag sweep (shared implementation for both interactive and CLI usage).

        Args:
            log_dir: Directory containing val_log.txt and checkpoints
            token_interval: Token interval in raw count (e.g., 500_000_000 for 500M)
            interactive: If True, prompt for confirmation before running
        """
        # Auto-detect save_step from checkpoint files
        save_step = self._detect_save_step(log_dir)
        if save_step is None:
            logger.print_and_log("Error: Could not auto-detect save step (need at least 2 checkpoints)")
            return
        logger.print_and_log(f"Auto-detected checkpoint save interval: {save_step} steps")

        # Parse val_log.txt to get step -> token mapping
        val_log_path = os.path.join(log_dir, "val_log.txt")
        if not os.path.exists(val_log_path):
            logger.print_and_log(f"Error: val_log.txt not found in {log_dir}")
            return

        step_to_tokens = {}
        with open(val_log_path, 'r') as f:
            for line in f:
                st_match = re.search(r'st:\s*(\d+)', line)
                tok_match = re.search(r'tok:\s*(\d+)', line)
                if st_match and tok_match:
                    step = int(st_match.group(1))
                    tokens = int(tok_match.group(1))
                    step_to_tokens[step] = tokens

        if not step_to_tokens:
            logger.print_and_log("Error: No step/token data found in val_log.txt")
            return

        logger.print_and_log(f"Parsed {len(step_to_tokens)} entries from val_log.txt")

        max_tokens = max(step_to_tokens.values())
        logger.print_and_log(f"Training range: 0 to {max_tokens/1e9:.2f}B tokens")

        # Check for existing results FIRST (before building evaluation list)
        hella_log_path = os.path.join(log_dir, "hellaswag_log.txt")
        evaluated_steps = set()
        max_evaluated_step = 0
        if os.path.exists(hella_log_path):
            with open(hella_log_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 1:
                        try:
                            step = int(parts[0].strip())
                            evaluated_steps.add(step)
                            max_evaluated_step = max(max_evaluated_step, step)
                        except ValueError:
                            pass
            logger.print_and_log(f"Found {len(evaluated_steps)} already-evaluated steps (max step: {max_evaluated_step})")

        # Calculate milestones
        milestones = []
        milestone = token_interval
        while milestone <= max_tokens:
            milestones.append(milestone)
            milestone += token_interval

        logger.print_and_log(f"Found {len(milestones)} token milestones total")

        # Find closest checkpoint for each milestone
        # For each milestone, find the best matching step overall. If that step is
        # already evaluated, the milestone is covered and we skip it. Only include
        # milestones whose best-match step hasn't been evaluated yet.
        steps_to_evaluate = []
        for milestone in milestones:
            best_step = None
            best_diff = float('inf')

            for step, tokens in step_to_tokens.items():
                if step % save_step != 0:
                    continue
                diff = abs(tokens - milestone)
                if diff < best_diff:
                    best_diff = diff
                    best_step = step

            if best_step is None:
                continue

            # If the best matching step was already evaluated, this milestone is covered
            if best_step in evaluated_steps:
                continue

            actual_tokens = step_to_tokens[best_step]
            steps_to_evaluate.append((best_step, actual_tokens, milestone))

        # Remove duplicates
        seen_steps = set()
        unique_steps = []
        for step, actual_tokens, target_milestone in steps_to_evaluate:
            if step not in seen_steps:
                seen_steps.add(step)
                unique_steps.append((step, actual_tokens, target_milestone))
        steps_to_evaluate = unique_steps

        if not steps_to_evaluate:
            logger.print_and_log("All milestones already evaluated!")
            return

        logger.print_and_log(f"Will evaluate {len(steps_to_evaluate)} checkpoints")
        for step, tokens, milestone in steps_to_evaluate:
            logger.print_and_log(f"  Step {step}: {tokens/1e6:.1f}M tokens (target: {milestone/1e6:.0f}M)")

        # Confirm before proceeding (only in interactive mode)
        if interactive:
            confirm = input("\nProceed with evaluation? (y/N): ")
            if confirm.lower() != 'y':
                logger.print_and_log("Evaluation cancelled.")
                return

        # Run evaluations
        results = []
        for i, (step, tokens, milestone) in enumerate(steps_to_evaluate):
            checkpoint_path = os.path.join(log_dir, f"model_step_{step:06d}.pt")

            if not os.path.exists(checkpoint_path):
                logger.print_and_log(f"[{i+1}/{len(steps_to_evaluate)}] Checkpoint not found: {checkpoint_path}, skipping...")
                continue

            logger.print_and_log(f"\n[{i+1}/{len(steps_to_evaluate)}] Evaluating step {step} ({tokens/1e6:.1f}M tokens)...")

            try:
                if self.model is not None:
                    del self.model
                    torch.cuda.empty_cache()

                self.model, self.enc, self.cfg = nc.load_model_and_tokenizer(
                    checkpoint_path,
                    device=self.device,
                    half_precision=self.half,
                    tok_kind=self.tok_kind,
                    tok_path=self.tok_path,
                    special_tokens=self.special_tokens,
                    shard_strategy=self.shard_strategy,
                    preferred_gpu=self.preferred_gpu,
                    max_memory_per_gpu=self.max_memory,
                    qk_norm_mode=self.qk_norm_mode,
                    use_keel=self.use_keel or None
                )

                num_correct, num_total = self._run_hellaswag_for_sweep()
                accuracy = 100.0 * num_correct / num_total

                logger.print_and_log(f"Step {step}: {accuracy:.2f}% ({num_correct}/{num_total})")

                with open(hella_log_path, 'a') as f:
                    f.write(f"{step}, {tokens}, {accuracy:.4f}\n")

                results.append((step, tokens, accuracy))

            except Exception as e:
                logger.print_and_log(f"Error evaluating step {step}: {str(e)}")
                continue

        # Summary
        logger.print_and_log(f"\n=== Evaluation Complete ===")
        logger.print_and_log(f"Results saved to: {hella_log_path}")
        for step, tokens, accuracy in results:
            logger.print_and_log(f"  Step {step} ({tokens/1e6:.1f}M tokens): {accuracy:.2f}%")

    # ----------------------- Coherence sweep -----------------------

    def cmd_coherence_sweep(self):
        """Interactive wrapper for run_coherence_sweep."""
        log_dir = input("Log directory path: ")
        if not log_dir or not os.path.isdir(log_dir):
            return f"Error: Invalid directory: {log_dir}"

        token_interval_str = input("Token interval in millions [500]: ") or "500"
        token_interval = int(token_interval_str) * 1_000_000

        gen_size_str = input("Generation length in tokens [512]: ") or "512"
        gen_size = int(gen_size_str)

        prompts_path = input("Prompt bank path [./coherence_prompts.json]: ") or "./coherence_prompts.json"

        self.run_coherence_sweep(
            log_dir=log_dir,
            token_interval=token_interval,
            gen_size=gen_size,
            prompts_path=prompts_path,
            interactive=True,
        )
        return ""

    def _run_coherence_for_sweep(self, prompts, gen_size, temperature, top_p, sweep_seed):
        """Run the prompt bank against the currently-loaded model and compute
        metrics. Returns (aggregate_dict, per_prompt_list)."""
        per_prompt = []
        context_size = self.model.params.max_seq_len

        t0 = time.time()
        for i, entry in enumerate(prompts):
            pid = entry["id"]
            prompt_text = entry["text"]
            # Per-prompt deterministic seed so sampling trajectories are
            # identical across checkpoints — any metric drift is the model.
            prompt_seed = (sweep_seed * 1000003 + hash(pid)) & 0x7FFFFFFF

            progress_prefix = f"  Prompt {i+1}/{len(prompts)} [{pid}]"

            result = nc.generate_with_stats(
                self.model, self.enc, prompt_text,
                max_new_tokens=gen_size,
                context_size=context_size,
                temperature=temperature,
                top_p=top_p,
                stop_on_eos=False,
                seed=prompt_seed,
                progress_prefix=progress_prefix,
            )
            metrics = cm.compute_all(
                result["text"],
                prompt_text=prompt_text,
                token_strings=result["token_strings"],
                per_token_entropy=result["per_token_entropy"],
            )
            per_prompt.append({
                "id": pid,
                "title": entry.get("title", ""),
                "stop_reason": result["stop_reason"],
                "tokens_generated": result["tokens_generated"],
                "text": result["text"],
                "metrics": metrics,
            })

            elapsed = time.time() - t0
            done = i + 1
            if done < len(prompts) and elapsed > 0:
                eta = elapsed * (len(prompts) - done) / done
                eta_m, eta_s = divmod(int(eta), 60)
                eta_str = f"sweep ETA {eta_m}m{eta_s:02d}s"
            else:
                eta_str = "done"
            # Overprint the live token-counter with the completion summary.
            print(
                f"\r  Prompt {done}/{len(prompts)} [{pid}] "
                f"done in {int(elapsed // 60)}m{int(elapsed % 60):02d}s total  {eta_str}"
                + " " * 30,
                end="",
            )
        print("")

        aggregate = cm.aggregate([p["metrics"] for p in per_prompt])
        return aggregate, per_prompt

    def run_coherence_sweep(self, log_dir, token_interval, gen_size=512,
                            prompts_path="./coherence_prompts.json",
                            temperature=0.7, top_p=0.9, sweep_seed=42,
                            interactive=False):
        """Run coherence-metric sweep across checkpoints in log_dir.

        Mirrors run_hella_sweep: auto-detects save_step from checkpoints,
        parses val_log.txt for step->tokens, walks token milestones, skips
        steps already in coherence_log.jsonl.
        """
        # Save step
        save_step = self._detect_save_step(log_dir)
        if save_step is None:
            logger.print_and_log("Error: Could not auto-detect save step (need at least 2 checkpoints)")
            return
        logger.print_and_log(f"Auto-detected checkpoint save interval: {save_step} steps")

        # Token map from val_log.txt
        val_log_path = os.path.join(log_dir, "val_log.txt")
        if not os.path.exists(val_log_path):
            logger.print_and_log(f"Error: val_log.txt not found in {log_dir}")
            return

        step_to_tokens = {}
        with open(val_log_path, 'r') as f:
            for line in f:
                st_match = re.search(r'st:\s*(\d+)', line)
                tok_match = re.search(r'tok:\s*(\d+)', line)
                if st_match and tok_match:
                    step_to_tokens[int(st_match.group(1))] = int(tok_match.group(1))

        if not step_to_tokens:
            logger.print_and_log("Error: No step/token data found in val_log.txt")
            return

        max_tokens = max(step_to_tokens.values())
        logger.print_and_log(f"Parsed {len(step_to_tokens)} entries from val_log.txt")
        logger.print_and_log(f"Training range: 0 to {max_tokens/1e9:.2f}B tokens")

        # Prompt bank
        if not os.path.isfile(prompts_path):
            logger.print_and_log(f"Error: prompt bank not found: {prompts_path}")
            return
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompt_bank = json.load(f)
        prompts = prompt_bank["prompts"]
        bank_version = prompt_bank.get("version", None)
        logger.print_and_log(f"Loaded {len(prompts)} prompts from {prompts_path} (version {bank_version})")

        # Existing results
        coh_log_path = os.path.join(log_dir, "coherence_log.jsonl")
        evaluated_steps = set()
        if os.path.exists(coh_log_path):
            with open(coh_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        evaluated_steps.add(int(rec["step"]))
                    except Exception:
                        pass
            logger.print_and_log(f"Found {len(evaluated_steps)} already-evaluated steps")

        # Milestone → closest save-step checkpoint
        milestones = []
        m = token_interval
        while m <= max_tokens:
            milestones.append(m)
            m += token_interval
        logger.print_and_log(f"Found {len(milestones)} token milestones total")

        steps_to_evaluate = []
        seen_steps = set()
        for milestone in milestones:
            best_step = None
            best_diff = float('inf')
            for step, tokens in step_to_tokens.items():
                if step % save_step != 0:
                    continue
                diff = abs(tokens - milestone)
                if diff < best_diff:
                    best_diff = diff
                    best_step = step
            if best_step is None or best_step in evaluated_steps or best_step in seen_steps:
                continue
            seen_steps.add(best_step)
            steps_to_evaluate.append((best_step, step_to_tokens[best_step], milestone))

        if not steps_to_evaluate:
            logger.print_and_log("All milestones already evaluated!")
            return

        logger.print_and_log(f"Will evaluate {len(steps_to_evaluate)} checkpoints")
        for step, tokens, milestone in steps_to_evaluate:
            logger.print_and_log(f"  Step {step}: {tokens/1e6:.1f}M tokens (target: {milestone/1e6:.0f}M)")

        if interactive:
            confirm = input("\nProceed with evaluation? (y/N): ")
            if confirm.lower() != 'y':
                logger.print_and_log("Evaluation cancelled.")
                return

        # Main loop
        for i, (step, tokens, milestone) in enumerate(steps_to_evaluate):
            checkpoint_path = os.path.join(log_dir, f"model_step_{step:06d}.pt")
            if not os.path.exists(checkpoint_path):
                logger.print_and_log(f"[{i+1}/{len(steps_to_evaluate)}] Missing: {checkpoint_path}, skipping")
                continue

            logger.print_and_log(f"\n[{i+1}/{len(steps_to_evaluate)}] Step {step} ({tokens/1e6:.1f}M tokens)")

            try:
                if self.model is not None:
                    del self.model
                    torch.cuda.empty_cache()

                self.model, self.enc, self.cfg = nc.load_model_and_tokenizer(
                    checkpoint_path,
                    device=self.device,
                    half_precision=self.half,
                    tok_kind=self.tok_kind,
                    tok_path=self.tok_path,
                    special_tokens=self.special_tokens,
                    shard_strategy=self.shard_strategy,
                    preferred_gpu=self.preferred_gpu,
                    max_memory_per_gpu=self.max_memory,
                    qk_norm_mode=self.qk_norm_mode,
                    use_keel=self.use_keel or None,
                )

                aggregate, per_prompt = self._run_coherence_for_sweep(
                    prompts, gen_size, temperature, top_p, sweep_seed,
                )

                record = {
                    "step": step,
                    "tokens": tokens,
                    "milestone": milestone,
                    "gen_size": gen_size,
                    "temperature": temperature,
                    "top_p": top_p,
                    "sweep_seed": sweep_seed,
                    "prompt_bank_version": bank_version,
                    "n_prompts": len(prompts),
                    "aggregate": aggregate,
                    "per_prompt": per_prompt,
                }

                with open(coh_log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record) + "\n")

                # Compact summary to console
                a = aggregate
                logger.print_and_log(
                    f"Step {step}: "
                    f"nonword={a.get('nonword_rate', 0):.4f}  "
                    f"new_ent_med={a.get('new_entities_introduced_median', 0):.1f}  "
                    f"xspan={a.get('cross_span_entity', 0):.3f}"
                )

            except Exception as e:
                logger.print_and_log(f"Error evaluating step {step}: {str(e)}")
                import traceback; traceback.print_exc()
                continue

        logger.print_and_log(f"\n=== Coherence Sweep Complete ===")
        logger.print_and_log(f"Results appended to: {coh_log_path}")
        try:
            from pathlib import Path as _Path
            redacted_path, n_recs, _ = _redact_coherence_log(_Path(coh_log_path))
            logger.print_and_log(f"Redacted log written to: {redacted_path}  ({n_recs} record(s))")
        except Exception as e:
            logger.print_and_log(f"Warning: failed to write redacted log: {e}")

    def run_coherence_step(self, log_dir, step, gen_size=512,
                           prompts_path="./coherence_prompts.json",
                           temperature=0.7, top_p=0.9, sweep_seed=42,
                           force=False):
        """Ad-hoc single-checkpoint coherence eval. Appends one record to
        coherence_log.jsonl in the same format as run_coherence_sweep.

        Skips silently if `step` is already in the log, unless force=True.
        """
        checkpoint_path = os.path.join(log_dir, f"model_step_{step:06d}.pt")
        if not os.path.exists(checkpoint_path):
            logger.print_and_log(f"Error: checkpoint not found: {checkpoint_path}")
            return

        # Token count from val_log.txt. Val is logged every N steps (not every
        # checkpoint save) so exact matches may miss — fall back to linear
        # interpolation between the nearest logged neighbors.
        tokens = None
        val_log_path = os.path.join(log_dir, "val_log.txt")
        if os.path.exists(val_log_path):
            step_to_tokens = {}
            with open(val_log_path, 'r') as f:
                for line in f:
                    st_match = re.search(r'st:\s*(\d+)', line)
                    tok_match = re.search(r'tok:\s*(\d+)', line)
                    if st_match and tok_match:
                        step_to_tokens[int(st_match.group(1))] = int(tok_match.group(1))
            if step in step_to_tokens:
                tokens = step_to_tokens[step]
            else:
                below = max((s for s in step_to_tokens if s < step), default=None)
                above = min((s for s in step_to_tokens if s > step), default=None)
                if below is not None and above is not None:
                    frac = (step - below) / (above - below)
                    tokens = int(step_to_tokens[below] + frac * (step_to_tokens[above] - step_to_tokens[below]))
                    logger.print_and_log(
                        f"Interpolated tokens for step {step}: {tokens} "
                        f"(from step {below}={step_to_tokens[below]} and step {above}={step_to_tokens[above]})"
                    )
        if tokens is None:
            logger.print_and_log(f"Warning: no token count for step {step} in val_log.txt")

        # Prompt bank.
        if not os.path.isfile(prompts_path):
            logger.print_and_log(f"Error: prompt bank not found: {prompts_path}")
            return
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompt_bank = json.load(f)
        prompts = prompt_bank["prompts"]
        bank_version = prompt_bank.get("version", None)

        # Skip-if-present check.
        coh_log_path = os.path.join(log_dir, "coherence_log.jsonl")
        if not force and os.path.exists(coh_log_path):
            with open(coh_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        if int(json.loads(line).get("step")) == step:
                            logger.print_and_log(
                                f"Step {step} already in {coh_log_path}. Use --force to re-evaluate."
                            )
                            return
                    except Exception:
                        pass

        logger.print_and_log(
            f"Evaluating step {step}"
            + (f" ({tokens/1e6:.1f}M tokens)" if tokens else "")
        )

        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()

        self.model, self.enc, self.cfg = nc.load_model_and_tokenizer(
            checkpoint_path,
            device=self.device,
            half_precision=self.half,
            tok_kind=self.tok_kind,
            tok_path=self.tok_path,
            special_tokens=self.special_tokens,
            shard_strategy=self.shard_strategy,
            preferred_gpu=self.preferred_gpu,
            max_memory_per_gpu=self.max_memory,
            qk_norm_mode=self.qk_norm_mode,
            use_keel=self.use_keel or None,
        )

        aggregate, per_prompt = self._run_coherence_for_sweep(
            prompts, gen_size, temperature, top_p, sweep_seed,
        )

        record = {
            "step": step,
            "tokens": tokens,
            "milestone": None,
            "gen_size": gen_size,
            "temperature": temperature,
            "top_p": top_p,
            "sweep_seed": sweep_seed,
            "prompt_bank_version": bank_version,
            "n_prompts": len(prompts),
            "aggregate": aggregate,
            "per_prompt": per_prompt,
        }
        with open(coh_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + "\n")

        a = aggregate
        logger.print_and_log(
            f"Step {step}: "
            f"nonword={a.get('nonword_rate', 0):.4f}  "
            f"new_ent_med={a.get('new_entities_introduced_median', 0):.1f}  "
            f"xspan={a.get('cross_span_entity', 0):.3f}"
        )
        logger.print_and_log(f"Appended to: {coh_log_path}")
        try:
            from pathlib import Path as _Path
            redacted_path, n_recs, _ = _redact_coherence_log(_Path(coh_log_path))
            logger.print_and_log(f"Redacted log written to: {redacted_path}  ({n_recs} record(s))")
        except Exception as e:
            logger.print_and_log(f"Warning: failed to write redacted log: {e}")

    def cmd_export(self):
        if not self.model_path or not self.model_path.endswith(".pt"):
            logger.print_and_log(f"Exporting only works with .pt models, not: {self.model_path}")
            exit(1)
        # Get the name from model_path, change it to .bin, and save it to cwd
        # self.model_path is guaranteed to be a string at this point
        export_name = os.path.basename(self.model_path).replace(".pt", ".bin")
        logger.print_and_log(f"Exporting model to: {export_name}")
        self.model.export(os.path.join("./", f"raw_{export_name}"))
        return f"Model exported to {export_name}"
    
    def cmd_mmlu(self):
        # for a default "all" test:
        subjects = None
        test_mmlu(self.model, self.enc, self.device, self.enc.pad_id, batch_size=self.eval_batch_size, subjects=subjects)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained model.")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to model checkpoint (.pt file) or directory (auto-selects most recent .pt file by step number)")
    parser.add_argument("--gen_size", type=str, default="1024", help="Size of the generated text.")
    parser.add_argument("--prompt_file", type=str, default="ev4.yaml", help="Path to the prompt file.")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling value.")
    parser.add_argument("--full", action="store_true", help="Use full precision (fp32) instead of half precision.")
    parser.add_argument("--tok_kind", type=str, default=None, help="Tokenizer kind: 'llama' or 'hf' (auto-detected from checkpoint if not specified).")
    parser.add_argument("--tok_path", type=str, default=None, help="Path to the tokenizer files (auto-detected from checkpoint if not specified).")
    parser.add_argument("--special_tokens", type=str, default=None, help="Path to special tokens JSON file (auto-detected from checkpoint if not specified).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluations (higher = faster but more memory).")
    parser.add_argument("--gpu", type=int, default=None, help="Specify which GPU to use (0, 1, etc). Use -1 for last GPU. Defaults to GPU 1 if available.")
    parser.add_argument("--shard_strategy", type=str, default="none", 
                        choices=["auto", "balanced", "none"], 
                        help="Sharding strategy: 'balanced' forces even split, 'auto' lets Accelerate decide, 'none' disables sharding")
    parser.add_argument("--max_memory", type=str, default=None, 
                        help="Max memory per GPU when sharding (e.g., '14GiB' for 4080s)")
    parser.add_argument("--qk_norm_mode", type=str, default=None,
                    choices=[None, "none", "before_rope", "after_rope_legacy", "after_rope_fixed"],
                    help="QK normalization mode: none (disabled), before_rope (learnable RMSNorm, recommended), "
                         "after_rope_legacy (L2 + sqrt(d) scaling), after_rope_fixed (L2 only)")
    parser.add_argument("--user", type=str, default="Josef, Joseph",)
    parser.add_argument("--use_keel", action="store_true",
                        help="Enable KEEL (Highway-style Post-LN) - use when checkpoint was trained with use_keel but config doesn't include it")

    # HellaSwag sweep arguments
    parser.add_argument("--hella_sweep", action="store_true",
                        help="Run HellaSwag sweep on checkpoints in --model_path directory (non-interactive mode)")
    parser.add_argument("--token_interval", type=int, default=500,
                        help="Token interval in millions for sweep milestones (default: 500)")

    # Coherence sweep arguments
    parser.add_argument("--coherence_sweep", action="store_true",
                        help="Run coherence-metric sweep on checkpoints in --model_path directory (non-interactive mode)")
    parser.add_argument("--coherence_prompts", type=str, default="./coherence_prompts.json",
                        help="Path to coherence prompt bank JSON (default: ./coherence_prompts.json)")
    parser.add_argument("--coherence_gen_size", type=int, default=512,
                        help="Tokens generated per prompt during coherence sweep (default: 512)")
    parser.add_argument("--coherence_temp", type=float, default=0.7,
                        help="Sampling temperature for coherence sweep (default: 0.7)")
    parser.add_argument("--coherence_top_p", type=float, default=0.9,
                        help="Top-p for coherence sweep (default: 0.9)")
    parser.add_argument("--coherence_seed", type=int, default=42,
                        help="Sweep seed for deterministic sampling across checkpoints (default: 42)")
    parser.add_argument("--coherence_step", type=int, default=None,
                        help="One-off coherence eval on a single checkpoint step. Appends to coherence_log.jsonl.")
    parser.add_argument("--coherence_force", action="store_true",
                        help="With --coherence_step, re-evaluate even if the step is already in the log.")
    return parser.parse_args()

# Main entry point
if __name__ == "__main__":

    # We have to shut off the logger FIRST
    logger.print_and_log("Model util v0.3 (simplified)")

    torch.backends.cuda.matmul.allow_tf32 = True     # if you're on Ada/Lovelace

    # Use A time-based seed
    seed = int(time.time())
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    args = parse_args()

    # Require model_path for all modes
    if not args.model_path:
        logger.print_and_log("Error: --model_path is required")
        sys.exit(1)

    # Check if running in hella_sweep mode (non-interactive)
    if args.hella_sweep:
        log_dir = args.model_path
        if not os.path.isdir(log_dir):
            logger.print_and_log(f"Error: --model_path must be a directory for --hella_sweep: {log_dir}")
            sys.exit(1)

        # Create a Generate instance without loading a model initially
        myGen = Generate("", preferred_gpu=args.gpu)
        myGen.half = not args.full
        myGen.eval_batch_size = args.batch_size
        myGen.tok_kind = args.tok_kind
        myGen.tok_path = args.tok_path
        myGen.shard_strategy = args.shard_strategy
        myGen.max_memory = args.max_memory
        qk_mode = getattr(args, 'qk_norm_mode', None)
        if qk_mode is not None:
            qk_mode = qk_mode.lower()
            if qk_mode == "none":
                qk_mode = None
        myGen.qk_norm_mode = qk_mode
        myGen.use_keel = args.use_keel

        # Run the sweep using the refactored function
        myGen.run_hella_sweep(
            log_dir=log_dir,
            token_interval=args.token_interval * 1_000_000
        )
        sys.exit(0)

    # Check if running in coherence_sweep mode (non-interactive)
    if args.coherence_sweep:
        log_dir = args.model_path
        if not os.path.isdir(log_dir):
            logger.print_and_log(f"Error: --model_path must be a directory for --coherence_sweep: {log_dir}")
            sys.exit(1)

        myGen = Generate("", preferred_gpu=args.gpu)
        myGen.half = not args.full
        myGen.eval_batch_size = args.batch_size
        myGen.tok_kind = args.tok_kind
        myGen.tok_path = args.tok_path
        myGen.shard_strategy = args.shard_strategy
        myGen.max_memory = args.max_memory
        qk_mode = getattr(args, 'qk_norm_mode', None)
        if qk_mode is not None:
            qk_mode = qk_mode.lower()
            if qk_mode == "none":
                qk_mode = None
        myGen.qk_norm_mode = qk_mode
        myGen.use_keel = args.use_keel
        myGen.special_tokens = args.special_tokens

        myGen.run_coherence_sweep(
            log_dir=log_dir,
            token_interval=args.token_interval * 1_000_000,
            gen_size=args.coherence_gen_size,
            prompts_path=args.coherence_prompts,
            temperature=args.coherence_temp,
            top_p=args.coherence_top_p,
            sweep_seed=args.coherence_seed,
        )
        sys.exit(0)

    # One-off coherence eval on a single step
    if args.coherence_step is not None:
        log_dir = args.model_path
        if not os.path.isdir(log_dir):
            logger.print_and_log(f"Error: --model_path must be a directory for --coherence_step: {log_dir}")
            sys.exit(1)

        myGen = Generate("", preferred_gpu=args.gpu)
        myGen.half = not args.full
        myGen.eval_batch_size = args.batch_size
        myGen.tok_kind = args.tok_kind
        myGen.tok_path = args.tok_path
        myGen.shard_strategy = args.shard_strategy
        myGen.max_memory = args.max_memory
        qk_mode = getattr(args, 'qk_norm_mode', None)
        if qk_mode is not None:
            qk_mode = qk_mode.lower()
            if qk_mode == "none":
                qk_mode = None
        myGen.qk_norm_mode = qk_mode
        myGen.use_keel = args.use_keel
        myGen.special_tokens = args.special_tokens

        myGen.run_coherence_step(
            log_dir=log_dir,
            step=args.coherence_step,
            gen_size=args.coherence_gen_size,
            prompts_path=args.coherence_prompts,
            temperature=args.coherence_temp,
            top_p=args.coherence_top_p,
            sweep_seed=args.coherence_seed,
            force=args.coherence_force,
        )
        sys.exit(0)

    # Normal interactive mode - resolve model path (handles both files and directories)
    resolved_path = resolve_model_path(args.model_path)

    myGen = Generate("", preferred_gpu=args.gpu)
    myGen.model_path = resolved_path
    myGen.gen_size = int(args.gen_size)
    myGen.prompt_file = args.prompt_file
    myGen.temp = args.temp
    myGen.top_p = args.top_p
    myGen.half = not args.full
    myGen.eval_batch_size = args.batch_size
    myGen.user = [name.strip() for name in args.user.split(",") if name.strip()]

    # Store the initial settings
    myGen.tok_kind = args.tok_kind
    myGen.tok_path = args.tok_path
    myGen.shard_strategy = args.shard_strategy
    myGen.max_memory = args.max_memory
    qk_mode = getattr(args, 'qk_norm_mode', None)
    if qk_mode is not None:
        qk_mode = qk_mode.lower()
        if qk_mode == "none":
            qk_mode = None
    myGen.qk_norm_mode = qk_mode
    myGen.use_keel = args.use_keel
    myGen.special_tokens = args.special_tokens

    # Load the model using neo_common
    start_time = time.time()
    myGen.model, myGen.enc, myGen.cfg = nc.load_model_and_tokenizer(
        myGen.model_path,
        device=myGen.device,
        half_precision=myGen.half,
        tok_kind=args.tok_kind,
        tok_path=args.tok_path,
        special_tokens=args.special_tokens,
        shard_strategy=args.shard_strategy,  # Pass sharding strategy
        preferred_gpu=args.gpu,
        max_memory_per_gpu=args.max_memory,  # e.g., "14GiB"
        qk_norm_mode=myGen.qk_norm_mode,
        use_keel=myGen.use_keel or None
    )

    logger.print_and_log(f"Model loaded in {time.time() - start_time:.2f} seconds")
    logger.print_and_log(f"Default evaluation batch size: {myGen.eval_batch_size} (use 'batch' command to change)")

    logger.print_and_log(f"Type 'help' for a list of commands.")

    while True:
        is_cmd, command = myGen.do_user_command()
        if not is_cmd:
            logger.print_and_log("Invalid command")
            continue