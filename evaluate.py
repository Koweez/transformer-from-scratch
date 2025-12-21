"""
GPT Model Evaluation and Text Generation.

This script provides efficient text generation using a trained GPT model
with KV-cache optimization for fast autoregressive decoding.

Key Features:
    - KV-Cache: Caches key-value pairs to avoid redundant computation
    - Temperature Sampling: Controls randomness in generation
    - Performance Metrics: Tracks tokens/second generation speed

KV-Cache Explanation:
    Without cache: Each new token requires recomputing attention over ALL previous tokens
    With cache: Only compute attention for the NEW token, reuse cached K,V from history
    
    Complexity improvement: O(n²) per generation → O(n) per generation

Usage:
    python evaluate.py

The script loads a checkpoint and generates Shakespeare-style text.
"""

import torch
import time
from transformers import AutoTokenizer
from models.transformer import GPTTransformer
from config import GPT_CONFIG as cfg


def load_model(checkpoint_path, device):
    """
    Load a trained GPT model and its tokenizer.
    
    Args:
        checkpoint_path (str): Path to the saved model weights (.pth file)
        device (str): Device to load model on ('cuda' or 'cpu')
    
    Returns:
        model (GPTTransformer): Loaded model in eval mode
        tokenizer (AutoTokenizer): Corresponding tokenizer
    """
    # Initialize tokenizer (must match training tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model architecture
    model = GPTTransformer(
        vocab_size=len(tokenizer),
        num_layers=cfg["num_layers"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        max_len=cfg["max_len"]
    ).to(device)
    
    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    return model, tokenizer


@torch.no_grad()
def generate_efficiently(model, tokenizer, prompt, max_new_tokens=100, temp=0.8, device="cuda"):
    """
    Generate text efficiently using KV-cache.
    
    This function implements a two-phase generation strategy:
    
    1. PRE-FILL PHASE: Process the entire prompt in one forward pass.
       This builds the initial KV-cache containing the prompt's context.
       Complexity: O(prompt_len²) - but only done ONCE.
    
    2. DECODE PHASE: Generate tokens one at a time, using cached K,V.
       Each step only processes 1 new token and attends to cached history.
       Complexity: O(prompt_len + generated_len) per token.
    
    Args:
        model (GPTTransformer): Trained model
        tokenizer (AutoTokenizer): Tokenizer for encoding/decoding
        prompt (str): Text prompt to continue from
        max_new_tokens (int): Maximum number of tokens to generate. Default: 100
        temp (float): Temperature for sampling (higher = more random). Default: 0.8
        device (str): Device for computation. Default: 'cuda'
    
    Returns:
        full_text (str): Complete generated text (prompt + generation)
        tokens_per_sec (float): Generation speed metric
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Start timing
    start_time = time.time()
    
    # =========================================================================
    # PHASE 1: PRE-FILL (Process prompt, build initial KV-cache)
    # =========================================================================
    # This is O(n²) but we only do it ONCE for the entire prompt
    logits, past_key_values = model(input_ids, layer_caches=None)
    
    # Sample first new token
    next_token_logits = logits[:, -1, :] / temp  # Temperature scaling
    probs = torch.softmax(next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)  # Sampling
    
    output_ids = [next_token.item()]
    
    # =========================================================================
    # PHASE 2: INCREMENTAL DECODING (Fast token-by-token generation)
    # =========================================================================
    # Only pass the NEW token; attention is computed over cached history
    for _ in range(max_new_tokens - 1):
        # Forward pass with only the new token + cached K,V
        logits, past_key_values = model(next_token, layer_caches=past_key_values)
        
        # Sample next token with temperature
        next_token_logits = logits[:, -1, :] / temp
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        token_id = next_token.item()
        output_ids.append(token_id)
        
        # Stop on EOS token
        if token_id == tokenizer.eos_token_id:
            break
            
    # Calculate generation speed
    total_time = time.time() - start_time
    tokens_per_sec = len(output_ids) / total_time
    
    # Decode full sequence (prompt + generated)
    full_text = tokenizer.decode(
        torch.cat([input_ids[0], torch.tensor(output_ids).to(device)]), 
        skip_special_tokens=True
    )
    
    return full_text, tokens_per_sec


def main():
    """
    Main evaluation function.
    
    Loads the best checkpoint and generates sample text to demonstrate
    the model's learned Shakespeare-style writing.
    """
    # Load trained model
    model, tokenizer = load_model(cfg["checkpoint_path"], cfg["device"])
    
    # Test prompt (Shakespeare character)
    prompt = "ROMEO: Shall I believe"
    print(f"\nPrompt: {prompt}")
    
    # Generate text with KV-cache
    generated_text, speed = generate_efficiently(model, tokenizer, prompt)
    
    # Display results
    print("-" * 30)
    print(generated_text)
    print("-" * 30)
    print(f"🚀 Speed: {speed:.2f} tokens/sec")


if __name__ == "__main__":
    main()
