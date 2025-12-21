"""
GPT Training Script.

This script trains a GPT-style language model on the Tiny Shakespeare dataset.
It includes:
- Automatic mixed precision (AMP) training for faster GPU utilization
- TensorBoard logging for monitoring training progress
- Best model checkpointing based on validation loss
- Efficient data loading with chunked tokenization

Usage:
    python train_gpt.py

Requirements:
    - PyTorch with CUDA support
    - HuggingFace transformers and datasets
    - TensorboardX for logging

The trained model can be evaluated using evaluate.py for text generation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import os

# Import custom model and configuration
from models.transformer import GPTTransformer
from config import GPT_CONFIG as cfg


def train():
    """
    Main training function for GPT language model.
    
    Training Pipeline:
        1. Setup logging and directories
        2. Load and tokenize dataset
        3. Initialize model, optimizer, and loss function
        4. Train with AMP and gradient scaling
        5. Save best checkpoint based on loss
    """
    
    # =========================================================================
    # 1. INITIALIZE LOGGING & ENVIRONMENT
    # =========================================================================
    os.makedirs(os.path.dirname(cfg["checkpoint_path"]), exist_ok=True)
    writer = SummaryWriter(logdir=cfg["tensorboard_log_dir"])
    device = cfg["device"]
    
    # =========================================================================
    # 2. LOAD DATASET & TOKENIZER
    # =========================================================================
    print("Loading Dataset and Tokenizer...")
    dataset_path = cfg["dataset_path"]
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    # Load Tiny Shakespeare from HuggingFace
    dataset = load_dataset(
        "karpathy/tiny_shakespeare", 
        revision="refs/convert/parquet",  # Force-load the parquet version
        split="train",
        cache_dir=os.path.dirname(dataset_path)
    )
    
    # Use Mistral tokenizer (32k vocab, BPE-based)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-style: use EOS as PAD
    
    def tokenize_fn(examples):
        """
        Tokenize and chunk text into fixed-length sequences.
        
        For language modeling, we concatenate all text and split into
        chunks of max_len tokens. This ensures efficient GPU utilization
        and consistent sequence lengths.
        """
        # Tokenize all texts without truncation
        tokenized = tokenizer(examples['text'], truncation=False)
        
        # Concatenate all token ids into one long sequence
        all_input_ids = []
        for ids in tokenized['input_ids']:
            all_input_ids.extend(ids)
        
        # Split into fixed-size chunks (drop incomplete final chunk)
        chunk_size = cfg["max_len"]
        chunks = []
        for i in range(0, len(all_input_ids) - chunk_size + 1, chunk_size):
            chunks.append(all_input_ids[i:i + chunk_size])
        
        return {"input_ids": chunks}
    
    # Apply tokenization and create DataLoader
    tokenized_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_ds.set_format("torch")
    loader = DataLoader(tokenized_ds, batch_size=cfg["batch_size"], shuffle=True)

    # =========================================================================
    # 3. MODEL SETUP
    # =========================================================================
    model = GPTTransformer(
        vocab_size=len(tokenizer),
        num_layers=cfg["num_layers"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        max_len=cfg["max_len"]
    ).to(device)

    # AdamW optimizer (decoupled weight decay for transformers)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg["lr"], 
        weight_decay=cfg["weight_decay"]
    )
    
    # Gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler("cuda")
    
    # Cross-entropy loss (ignoring padding tokens)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    best_loss = float('inf')

    # =========================================================================
    # 4. TRAINING LOOP
    # =========================================================================
    print(f"Starting Training on {device}...")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(cfg["epochs"]):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        epoch_loss = 0

        for i, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            labels = input_ids.clone()  # For LM, labels = inputs (shifted internally)

            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.amp.autocast("cuda"):
                logits = model(input_ids)
                
                # Causal Language Modeling: predict token[t+1] from token[0:t]
                # Shift logits and labels for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()  # Predictions for positions 0 to N-1
                shift_labels = labels[:, 1:].contiguous()       # Targets for positions 1 to N
                
                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log to TensorBoard every 10 steps
            if i % 10 == 0:
                step = epoch * len(loader) + i
                writer.add_scalar('Batch/Loss', loss.item(), step)

        # End of epoch statistics
        avg_loss = epoch_loss / len(loader)
        writer.add_scalar('Epoch/Avg_Loss', avg_loss, epoch)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")

        # Save best model checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), cfg["checkpoint_path"])
            print(f"--> Saved New Best Model to {cfg['checkpoint_path']}")

    writer.close()
    print("Training Finished Successfully.")


if __name__ == "__main__":
    train()
