"""
Encoder Architecture for Transformer Models.

This module implements the encoder component for encoder-decoder transformers:
- EncoderLayer: Single transformer encoder block
- Encoder: Full encoder stack

The encoder processes the source sequence and produces contextualized
representations used by the decoder for cross-attention.

References:
    - "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
from .layers import LayerNorm, FeedForward, TokenEmbedding, PositionalEncoding
from .attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer (Post-LayerNorm).
    
    Contains two sub-layers:
    1. Multi-head self-attention
    2. Position-wise feed-forward network
    
    Each sub-layer has a residual connection followed by layer normalization:
        output = LayerNorm(x + Sublayer(x))
    
    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads. Default: 8
        d_ff (int): Feed-forward hidden dimension. Default: 2048
        dropout (float): Dropout probability. Default: 0.1
    """
    
    def __init__(self, d_model, num_heads=8, d_ff=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, 
                                                       num_heads=num_heads, dropout=dropout)
        self.layer_norm1 = LayerNorm(d_model=d_model)
        self.layer_norm2 = LayerNorm(d_model=d_model)
        self.ff1 = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.ff2 = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Process input through encoder layer.
        
        Args:
            x (Tensor): Input of shape (batch, seq_len, d_model)
        
        Returns:
            Tensor: Encoder layer output of shape (batch, seq_len, d_model)
        """
        # 1. Multi-Head Self-Attention
        save = x
        weighted_output = self.multi_head_attention(x, x, x)
        out = weighted_output + save  # Residual connection
        out = self.layer_norm1(out)
        
        # 2. Feed-Forward Network
        save = out
        out = self.ff1(out)
        out = self.relu(out)
        out = self.ff2(out)
        out = out + save  # Residual connection
        out = self.layer_norm2(out)
        
        return out


class Encoder(nn.Module):
    """
    Full Transformer Encoder Stack.
    
    Processes source sequence through:
    1. Token embedding (with sqrt(d_model) scaling)
    2. Positional encoding
    3. Stack of N encoder layers
    
    Produces contextualized representations where each position
    can attend to all other positions in the sequence.
    
    Args:
        num_layers (int): Number of encoder layers (N)
        vocab_size (int): Source vocabulary size
        d_model (int): Model dimension
        num_heads (int): Number of attention heads. Default: 8
        d_ff (int): Feed-forward hidden dimension. Default: 2048
        dropout (float): Dropout probability. Default: 0.1
        max_len (int): Maximum sequence length. Default: 10000
    """
    
    def __init__(self, num_layers, vocab_size, d_model, 
                 num_heads=8, d_ff=2048, dropout=0.1, max_len=10000):
        super(Encoder, self).__init__()
        
        self.num_layers = num_layers
        self.embedding = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        
        # Stack of encoder layers
        self.encoders = nn.ModuleList([
            EncoderLayer(d_model=d_model, num_heads=num_heads, 
                        d_ff=d_ff, dropout=dropout) 
            for _ in range(num_layers)
        ])
    
    def forward(self, text):
        """
        Encode source sequence.
        
        Args:
            text (Tensor): Source token indices of shape (batch, src_len)
        
        Returns:
            Tensor: Encoder output of shape (batch, src_len, d_model)
        """
        # Embed and add positional information
        x = self.embedding(text)
        x = self.pos_encoder(x)
        
        # Process through encoder stack
        for layer in self.encoders:
            x = layer(x)
        
        return x
