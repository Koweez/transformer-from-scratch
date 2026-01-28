"""
KV-Cache Benchmark for GPT Transformer.

This script compares generation speed with and without KV-caching to demonstrate
the performance benefits of caching key-value pairs during autoregressive generation.

Without KV-Cache (O(N²) per generation):
    - Each new token requires recomputing attention over ALL previous tokens
    - Total complexity: O(1² + 2² + 3² + ... + N²) = O(N³)

With KV-Cache (O(N) per generation):
    - Previous K,V pairs are cached and reused
    - Each new token only computes its own K,V and attends to cached history
    - Total complexity: O(1 + 2 + 3 + ... + N) = O(N²)

Expected results: Speedup increases with sequence length as more redundant
computation is avoided by the cache.
"""

import time
import torch
from models.transformer import GPTTransformer
from transformers import AutoTokenizer
from evaluate import load_model
from typing import List


def benchmark_both_methods(model: GPTTransformer, tokenizer: AutoTokenizer, prompt: str, max_new_tokens: List[int]):
    """
    Benchmark generation with and without KV-cache for various sequence lengths.
    
    Uses model.generate() for both cases to ensure a fair comparison.
    
    Args:
        model: The GPT model to benchmark
        tokenizer: Tokenizer for encoding the prompt
        prompt: Text prompt to continue from
        max_new_tokens: List of token counts to test (e.g., [100, 200, 300])
    """
    # Tokenize prompt once (reused for all tests)
    tokenized = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    
    for max_len in max_new_tokens:
        print(f"\n=== Benchmarking generation for {max_len} new tokens ===")

        # ----- NO CACHE: Recompute full attention each step -----
        print("🏎️  NO CACHE (O(N²))")
        torch.cuda.synchronize()  # Ensure all GPU operations are complete
        start = time.time()
        with torch.no_grad():
            no_cache_output = model.generate(tokenized, max_len=max_len, use_cache=False)
        torch.cuda.synchronize()  # Wait for GPU to finish
        no_cache_time = time.time() - start
        print(f"⏱️  Time taken: {no_cache_time:.2f} seconds")
        
        # ----- KV CACHE: Reuse cached key-value pairs -----
        print("⚡ KV CACHE (O(N))")
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            cache_output = model.generate(tokenized, max_len=max_len, use_cache=True)
        torch.cuda.synchronize()
        cache_time = time.time() - start
        print(f"⏱️  Time taken: {cache_time:.2f} seconds")
        
        # ----- Results -----
        speedup = no_cache_time / cache_time
        print(f"🚀 SPEEDUP: {speedup:.1f}x")
        print(f"📊 Tokens/sec (cached): {max_len/cache_time:.0f}")


# =============================================================================
# Main execution
# =============================================================================

# Initialize tokenizer (using Mistral's tokenizer for vocabulary compatibility)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token  # GPT-style: use EOS as PAD

# Load trained model
model, _ = load_model("checkpoints/gpt_best.pth", device="cuda")

# Shakespeare-style prompt for generation
prompt = "To be, or not to be, that is the question:\n"

# Token counts to benchmark
# Note: Model max context is 512 tokens. Tests exceeding this will be capped.
max_new_tokens = [100, 200, 300, 500] # Last test is above max context because of prompt length (prompt + new tokens)

# Run benchmark
benchmark_both_methods(model, tokenizer, prompt, max_new_tokens)