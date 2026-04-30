#!/usr/bin/env python3
"""
Test script to verify KV cache implementation in chat_neo.py
Run with: python test_kv_cache.py
"""

import subprocess
import time
import sys

def run_generation_test(model_path, use_kv_cache=True):
    """Run a generation test with or without KV cache"""

    cmd = [
        sys.executable, "chat_neo.py",
        "--model_path", model_path,
        "--max_tokens", "50",
        "--temp", "0.7",
        "--force"  # Force response mode for testing
    ]

    if not use_kv_cache:
        cmd.append("--no_kv_cache")

    test_prompt = "Once upon a time"

    print(f"\n{'='*60}")
    print(f"Testing {'WITH' if use_kv_cache else 'WITHOUT'} KV cache")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Run the command with the test prompt
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(input=test_prompt, timeout=30)

        elapsed_time = time.time() - start_time

        print(f"Time taken: {elapsed_time:.2f} seconds")

        if process.returncode != 0:
            print(f"Error: {stderr}")
            return False, elapsed_time

        return True, elapsed_time

    except subprocess.TimeoutExpired:
        print("Test timed out after 30 seconds")
        process.kill()
        return False, 30.0
    except Exception as e:
        print(f"Error running test: {e}")
        return False, 0.0

def main():
    # You'll need to update this path to your actual model
    model_path = input("Enter path to your model checkpoint: ").strip()

    if not model_path:
        print("No model path provided. Exiting.")
        return

    print(f"\nTesting model: {model_path}")

    # Test without KV cache (disabled mode)
    success1, time1 = run_generation_test(model_path, use_kv_cache=False)

    # Test with KV cache (default mode)
    success2, time2 = run_generation_test(model_path, use_kv_cache=True)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if success1 and success2:
        print(f"Without KV cache: {time1:.2f}s")
        print(f"With KV cache:    {time2:.2f}s")

        if time2 < time1:
            speedup = (time1 / time2 - 1) * 100
            print(f"\nKV cache is {speedup:.1f}% faster!")
        else:
            print(f"\nKV cache was not faster in this test")
    else:
        if not success1:
            print("Test without KV cache failed")
        if not success2:
            print("Test with KV cache failed")

    print("\nNote: KV cache speedup is most noticeable with longer generations")
    print("and becomes more significant as the sequence length increases.")

if __name__ == "__main__":
    main()