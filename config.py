"""
Configuration Settings for Transformer Models.

This module contains hyperparameter configurations for:
- CONFIG: Standard encoder-decoder transformer (for translation tasks)
- GPT_CONFIG: GPT-style decoder-only transformer (for language modeling)

Paths are loaded from environment variables (.env file) to keep
sensitive/local information out of version control.

Setup:
    1. Copy .env.example to .env
    2. Edit .env with your local paths
    3. Install python-dotenv: pip install python-dotenv
"""

import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# ENCODER-DECODER TRANSFORMER CONFIG (for Seq2Seq tasks like translation)
# =============================================================================
CONFIG = {
    # Model Architecture
    'd_model': 512,              # Model/embedding dimension
    'num_heads': 8,              # Number of attention heads
    'num_encoder_layers': 6,     # Number of encoder blocks
    'num_decoder_layers': 6,     # Number of decoder blocks
    'd_ff': 2048,                # Feed-forward hidden dimension (typically 4x d_model)
    'dropout': 0.1,              # Dropout probability
    
    # Training
    'warming_up_steps': 4000     # Learning rate warmup steps
}

# =============================================================================
# GPT-STYLE DECODER-ONLY CONFIG (for Language Modeling)
# =============================================================================
GPT_CONFIG = {
    # --- System Settings (loaded from .env) ---
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "tensorboard_log_dir": os.getenv("TENSORBOARD_LOG_DIR", "./runs/shakespeare_experiment"),
    "checkpoint_path": os.getenv("CHECKPOINT_PATH", "./checkpoints/gpt_best.pth"),
    
    # --- Model Architecture ---
    # Following GPT-2 small-ish configuration
    "vocab_size": 32000,         # Will be updated by tokenizer length
    "num_layers": 12,            # Number of transformer blocks
    "d_model": 768,              # Hidden size / embedding dimension
    "num_heads": 12,             # Number of attention heads (d_model / 64 = head_dim)
    "d_ff": 3072,                # FFN intermediate size (4x d_model)
    "dropout": 0.1,              # Dropout probability
    "max_len": 512,              # Maximum context window length
    
    # --- Training Hyperparameters ---
    "epochs": 50,                # Number of training epochs
    "batch_size": 16,            # Batch size (adjust based on VRAM - 16 is safe for 16GB)
    "lr": 3e-4,                  # Learning rate (AdamW default for transformers)
    "weight_decay": 0.01,        # L2 regularization strength
    
    # --- Dataset Configuration (loaded from .env) ---
    "dataset_url": "https://huggingface.co/datasets/karpathy/tiny_shakespeare/resolve/main/default/train-00000-of-00001.parquet",
    "dataset_path": os.getenv("DATASET_PATH", "./data/shakespeare.parquet"),
}