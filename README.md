# Transformer from Scratch

A complete GPT-style transformer implementation with KV-cache optimization, built from scratch in PyTorch.

## Features

- **Decoder-only GPT architecture** with modern techniques (RMSNorm, SwiGLU, Pre-LayerNorm)
- **KV-Cache for efficient inference** - O(N) per token instead of O(N²)
- **Encoder-Decoder Transformer** for sequence-to-sequence tasks
- **TensorBoard logging** and checkpoint management
- Trained on **Tiny Shakespeare dataset**

## Project Structure

```
├── models/
│   ├── transformer.py   # GPTTransformer & Transformer (Seq2Seq)
│   ├── decoder.py       # GPTDecoder, GPTDecoderLayer
│   ├── encoder.py       # Encoder for Seq2Seq
│   ├── attention.py     # MultiHeadAttention with KV-cache
│   └── layers.py        # RMSNorm, SwiGLU, PositionalEncoding
├── train_gpt.py         # Training script for GPT
├── evaluate.py          # Text generation with KV-cache
├── bench_kv_cache.py    # KV-cache benchmark
├── config.py            # Model hyperparameters
└── checkpoints/         # Saved model weights
```

## Quick Start

```bash
# Train the model
python train_gpt.py

# Generate Shakespeare-style text
python evaluate.py

# Benchmark KV-cache speedup
python bench_kv_cache.py
```

## KV-Cache Benchmark Results

| Tokens | No Cache | KV Cache | Speedup |
|--------|----------|----------|---------|
| 100    | 0.90s    | 0.62s    | 1.5x    |
| 200    | 1.66s    | 1.25s    | 1.3x    |
| 300    | 2.71s    | 1.86s    | 1.5x    |
| 500    | 5.14s    | 3.05s    | 1.7x    |

Speedup increases with sequence length as more redundant computation is avoided.

## Model Configuration

```python
GPT_CONFIG = {
    "num_layers": 12,
    "d_model": 768,
    "num_heads": 12,
    "d_ff": 3072,
    "max_len": 512,
    "dropout": 0.1
}
```

## Architecture

```
Input → TokenEmbedding → PositionalEncoding
    ↓
┌─────────────────────────────────────┐
│  GPTDecoderLayer (x12)              │
│  ├── RMSNorm → MultiHeadAttention   │  ← KV-Cache here
│  └── RMSNorm → SwiGLU FFN           │
└─────────────────────────────────────┘
    ↓
Output Projection → Logits
```

## Training Results

- **Dataset**: Tiny Shakespeare (~1MB)
- **Final Loss**: ~1.8
- **Training**: 50 epochs on RTX 5070 Ti

View training curves: `tensorboard --logdir=runs/`
