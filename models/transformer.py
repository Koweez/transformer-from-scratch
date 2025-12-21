"""
Complete Transformer Model Architectures.

This module provides full transformer implementations:
- Transformer: Encoder-Decoder for sequence-to-sequence tasks (translation, summarization)
- GPTTransformer: Decoder-only for autoregressive language modeling

Both models support efficient inference with KV-caching for generation.

References:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
"""

import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder, GPTDecoder


class Transformer(nn.Module):
    """
    Encoder-Decoder Transformer for Sequence-to-Sequence tasks.
    
    Architecture:
        Source → Encoder → Context
        Target → Decoder(Context) → Output Probabilities
    
    Used for tasks like:
        - Machine Translation (e.g., English → French)
        - Text Summarization
        - Question Answering
    
    Args:
        src_vocab_size (int): Source vocabulary size
        tgt_vocab_size (int): Target vocabulary size
        num_encoder_layers (int): Number of encoder layers. Default: 6
        num_decoder_layers (int): Number of decoder layers. Default: 6
        d_model (int): Model dimension. Default: 512
        num_heads (int): Number of attention heads. Default: 8
        d_ff (int): Feed-forward hidden dimension. Default: 2048
        dropout (float): Dropout probability. Default: 0.1
        max_len (int): Maximum sequence length. Default: 10000
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, num_encoder_layers=6, 
                 num_decoder_layers=6, d_model=512, num_heads=8, 
                 d_ff=2048, dropout=0.1, max_len=10000):
        super(Transformer, self).__init__()
        
        # Encoder: processes source sequence
        self.encoder = Encoder(
            num_layers=num_encoder_layers, vocab_size=src_vocab_size, 
            d_model=d_model, num_heads=num_heads, d_ff=d_ff, 
            dropout=dropout, max_len=max_len
        )
        
        # Decoder: generates target sequence with cross-attention to encoder
        self.decoder = Decoder(
            num_layers=num_decoder_layers, vocab_size=tgt_vocab_size,
            d_model=d_model, num_heads=num_heads, d_ff=d_ff, 
            dropout=dropout, max_len=max_len
        )
        
        # Output projection to vocabulary
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt):
        """
        Full forward pass through encoder-decoder.
        
        Args:
            src (Tensor): Source tokens of shape (batch, src_len)
            tgt (Tensor): Target tokens of shape (batch, tgt_len)
        
        Returns:
            Tensor: Logits of shape (batch, tgt_len, tgt_vocab_size)
        """
        encoder_out = self.encoder(src)
        decoder_out = self.decoder(tgt, encoder_out)
        output = self.output_layer(decoder_out)
        return output
    
    def encode(self, src):
        """Encode source sequence."""
        return self.encoder(src)
    
    def decode(self, tgt, encoder_out):
        """Decode target sequence given encoder output."""
        return self.decoder(tgt, encoder_out)
    
    def generate(self, src, max_len, start_symbol):
        """
        Autoregressive generation for Seq2Seq.
        
        Uses greedy decoding (argmax at each step).
        
        Args:
            src (Tensor): Source tokens of shape (batch, src_len)
            max_len (int): Maximum generation length
            start_symbol (int): Start-of-sequence token ID
        
        Returns:
            Tensor: Generated token IDs of shape (batch, max_len)
        """
        # Encode source once
        encoder_out = self.encode(src)
        batch_size = src.size(0)
        
        # Start with <BOS> token
        ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src.data)
        
        # Generate tokens autoregressively
        for _ in range(max_len - 1):
            out = self.decode(ys, encoder_out)
            prob = self.output_layer(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)
        
        return ys


class GPTTransformer(nn.Module):
    """
    GPT-style Decoder-Only Transformer for Language Modeling.
    
    A decoder-only architecture for autoregressive language modeling.
    Predicts the next token given all previous tokens.
    
    Features:
        - Pre-LayerNorm architecture for stable training
        - KV-Cache support for efficient generation
        - SwiGLU activation in feed-forward layers
        - RMSNorm for faster normalization
    
    Architecture:
        Input Tokens → Embedding → Positional Encoding → 
        N × (Self-Attention + FFN) → Output Projection → Logits
    
    Args:
        vocab_size (int): Vocabulary size
        num_layers (int): Number of decoder layers. Default: 12
        d_model (int): Model dimension. Default: 768
        num_heads (int): Number of attention heads. Default: 12
        d_ff (int): Feed-forward hidden dimension. Default: 3072
        dropout (float): Dropout probability. Default: 0.1
        max_len (int): Maximum context length. Default: 512
    """
    
    def __init__(self, vocab_size, num_layers=12, d_model=768, 
                 num_heads=12, d_ff=3072, dropout=0.1, max_len=512):
        super(GPTTransformer, self).__init__()
        
        # GPT decoder with KV-cache support
        self.decoder = GPTDecoder(
            num_layers=num_layers, vocab_size=vocab_size,
            d_model=d_model, num_heads=num_heads, d_ff=d_ff, 
            dropout=dropout, max_len=max_len
        )
        
        # Output projection (language modeling head)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, tgt, layer_caches=None):
        """
        Forward pass with optional KV-cache.
        
        Args:
            tgt (Tensor): Input tokens of shape (batch, seq_len)
            layer_caches (list, optional): List of (past_key, past_value) tuples
                                          for each layer. None for full computation.
        
        Returns:
            output (Tensor): Logits of shape (batch, seq_len, vocab_size)
            presents (list): Updated KV-cache for next generation step
        """
        decoder_out, presents = self.decoder(tgt, layer_caches=layer_caches)
        output = self.output_layer(decoder_out)
        return output, presents
    
    def generate(self, src, max_len, start_symbol):
        """
        Autoregressive generation with KV-cache.
        
        Uses greedy decoding with caching for efficient generation.
        Note: This is a basic implementation. For production, see
        evaluate.py for temperature-based sampling.
        
        Args:
            src (Tensor): Context tokens (unused, for API compatibility)
            max_len (int): Maximum generation length
            start_symbol (int): Start token ID
        
        Returns:
            Tensor: Generated token IDs of shape (batch, max_len + 1)
        """
        batch_size = src.size(0)
        
        # Initialize with start token
        ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src.data)
        past_key_values = None
        
        # First forward pass (prefill)
        logits, past_key_values = self.forward(ys, layer_caches=past_key_values)
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
        
        # Incremental generation with KV-cache
        for _ in range(max_len - 1):
            logits, past_key_values = self.forward(ys, layer_caches=past_key_values)
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
        
        return ys