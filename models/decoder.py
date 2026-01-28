"""
Decoder Architectures for Transformer Models.

This module implements decoder components for both encoder-decoder (Seq2Seq)
and decoder-only (GPT-style) transformer architectures:
- DecoderLayer: Standard transformer decoder with cross-attention
- GPTDecoderLayer: Pre-LayerNorm decoder for GPT-style models
- Decoder: Full decoder stack for Seq2Seq models
- GPTDecoder: Decoder-only stack with KV-cache support

References:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .layers import FeedForward, GPTFeedForward, LayerNorm, RMSNorm, TokenEmbedding, PositionalEncoding


class DecoderLayer(nn.Module):
    """
    Standard Transformer Decoder Layer (Post-LayerNorm).
    
    Contains three sub-layers:
    1. Masked self-attention (causal)
    2. Cross-attention over encoder output
    3. Position-wise feed-forward network
    
    Each sub-layer has a residual connection followed by layer normalization.
    Used in encoder-decoder architectures for tasks like machine translation.
    
    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads. Default: 8
        d_ff (int): Feed-forward hidden dimension. Default: 2048
        dropout (float): Dropout probability. Default: 0.1
    """
    
    def __init__(self, d_model, num_heads=8, d_ff=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # Multi-head attention (used for both self and cross attention)
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, 
                                                       num_heads=num_heads, dropout=dropout)
        
        # Layer normalization for each sub-layer
        self.layer_norm1 = LayerNorm(d_model=d_model)
        self.layer_norm2 = LayerNorm(d_model=d_model)
        self.layer_norm3 = LayerNorm(d_model=d_model)
        
        # Feed-forward network
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    
    def forward(self, x, encoder_out, mask):
        """
        Process input through the decoder layer.
        
        Args:
            x (Tensor): Decoder input of shape (batch, tgt_len, d_model)
            encoder_out (Tensor): Encoder output of shape (batch, src_len, d_model)
            mask (Tensor): Causal attention mask
        
        Returns:
            Tensor: Decoder layer output of shape (batch, tgt_len, d_model)
        """
        # 1. Masked Self-Attention (decoder attends to previous positions only)
        save = x
        weighted_output = self.multi_head_attention(x, x, x, mask)
        out = weighted_output + save  # Residual connection
        out = self.layer_norm1(out)
        
        # 2. Cross-Attention (decoder attends to encoder output)
        save = out
        weighted_output = self.multi_head_attention(out, encoder_out, encoder_out, mask)
        out = weighted_output + save  # Residual connection
        out = self.layer_norm2(out)
        
        # 3. Feed-Forward Network
        save = out
        out = self.ff(out)
        out = out + save  # Residual connection
        out = self.layer_norm3(out)
        
        return out
    

class GPTDecoderLayer(nn.Module):
    """
    GPT-style Decoder Layer (Pre-LayerNorm).
    
    Uses pre-normalization (norm before attention/FFN) which has been shown
    to improve training stability. Includes scaled residual connections
    to prevent gradient explosion in deep models.
    
    Architecture:
        x -> LayerNorm -> Self-Attention -> Residual -> LayerNorm -> FFN -> Residual
    
    Args:
        d_model (int): Model dimension
        num_layers (int): Total number of layers (for residual scaling)
        num_heads (int): Number of attention heads. Default: 8
        d_ff (int): Feed-forward hidden dimension. Default: 2048
        dropout (float): Dropout probability. Default: 0.1
    """
    
    def __init__(self, d_model, num_layers, num_heads=8, d_ff=2048, dropout=0.1):
        super(GPTDecoderLayer, self).__init__()
        
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, 
                                                       num_heads=num_heads, dropout=dropout)
        self.norm1 = RMSNorm(d_model=d_model)
        self.norm2 = RMSNorm(d_model=d_model)
        self.ff = GPTFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        
        # Residual scaling factor: 1/sqrt(num_layers) to stabilize deep models
        self.factor = 1 / (num_layers ** 0.5)
    
    def forward(self, x, mask=None, layer_cache=None, use_cache=True):
        """
        Process input through the GPT decoder layer.
        
        Args:
            x (Tensor): Input of shape (batch, seq_len, d_model)
            mask (Tensor, optional): Causal mask (None during cached generation)
            layer_cache (tuple, optional): (past_key, past_value) for KV-cache
        
        Returns:
            out (Tensor): Layer output of shape (batch, seq_len, d_model)
            present (tuple): Current (key, value) for caching
        """
        # Pre-LayerNorm Self-Attention
        norm_x = self.norm1(x)
        
        if not use_cache:
            attn_out, present = self.multi_head_attention(
                norm_x, norm_x, norm_x, 
                mask=mask,
                layer_cache=None)
        else:
            attn_out, present = self.multi_head_attention(
                norm_x, norm_x, norm_x, 
                mask=mask,
                layer_cache=layer_cache)
        
        # Scaled residual connection
        out = x + attn_out * self.factor
        
        # Pre-LayerNorm Feed-Forward
        norm_out = self.norm2(out)
        ff_out = self.ff(norm_out)
        
        # Scaled residual connection
        out = out + ff_out * self.factor
        
        return out, present


class Decoder(nn.Module):
    """
    Full Transformer Decoder Stack (for Encoder-Decoder models).
    
    Stacks multiple DecoderLayers with token embedding and positional encoding.
    Used in Seq2Seq models for tasks like translation.
    
    Args:
        num_layers (int): Number of decoder layers
        vocab_size (int): Target vocabulary size
        d_model (int): Model dimension
        num_heads (int): Number of attention heads. Default: 8
        d_ff (int): Feed-forward hidden dimension. Default: 2048
        dropout (float): Dropout probability. Default: 0.1
        max_len (int): Maximum sequence length. Default: 10000
    """
    
    def __init__(self, num_layers, vocab_size, d_model, 
                 num_heads=8, d_ff=2048, dropout=0.1, max_len=10000):
        super(Decoder, self).__init__()
        
        self.num_layers = num_layers
        self.embedding = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        
        # Stack of decoder layers
        self.decoders = nn.ModuleList([
            DecoderLayer(d_model=d_model, num_heads=num_heads, 
                        d_ff=d_ff, dropout=dropout) 
            for _ in range(num_layers)
        ])
    
    def _create_causal_mask(self, seq_len, device):
        """
        Create causal (autoregressive) attention mask.
        
        Lower triangular matrix that prevents positions from attending
        to future positions (ensures left-to-right generation).
        
        Args:
            seq_len (int): Sequence length
            device: Torch device
        
        Returns:
            Tensor: Boolean mask of shape (1, 1, seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(1)  # Add batch and head dimensions
    
    def forward(self, text, encoder_out):
        """
        Process target sequence through decoder stack.
        
        Args:
            text (Tensor): Target token indices of shape (batch, tgt_len)
            encoder_out (Tensor): Encoder output of shape (batch, src_len, d_model)
        
        Returns:
            Tensor: Decoder output of shape (batch, tgt_len, d_model)
        """
        x = self.embedding(text)
        x = self.pos_encoder(x)
        mask = self._create_causal_mask(x.size(1), x.device)
        
        for layer in self.decoders:
            x = layer(x, encoder_out, mask)
        
        return x


class GPTDecoder(nn.Module):
    """
    GPT-style Decoder Stack with KV-Cache Support.
    
    Decoder-only transformer architecture for autoregressive language modeling.
    Supports efficient incremental generation using KV-cache.
    
    KV-Cache Mechanism:
        During generation, previously computed key-value pairs are cached.
        New tokens only need to compute their own K, V and attend to the
        full cached history, reducing complexity from O(n²) to O(n) per step.
    
    Args:
        num_layers (int): Number of decoder layers
        vocab_size (int): Vocabulary size
        d_model (int): Model dimension
        num_heads (int): Number of attention heads. Default: 8
        d_ff (int): Feed-forward hidden dimension. Default: 2048
        dropout (float): Dropout probability. Default: 0.1
        max_len (int): Maximum sequence length (context window). Default: 10000
    """
    
    def __init__(self, num_layers, vocab_size, d_model, 
                 num_heads=8, d_ff=2048, dropout=0.1, max_len=10000):
        super(GPTDecoder, self).__init__()
        
        self.embedding = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        
        # Stack of GPT decoder layers
        self.decoders = nn.ModuleList([
            GPTDecoderLayer(d_model=d_model, num_layers=num_layers, num_heads=num_heads, 
                           d_ff=d_ff, dropout=dropout) 
            for _ in range(num_layers)
        ])
    
    def _create_causal_mask(self, seq_len, device):
        """
        Create causal attention mask.
        
        Args:
            seq_len (int): Sequence length
            device: Torch device
        
        Returns:
            Tensor: Boolean mask of shape (1, 1, seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(1)
    
    def forward(self, text, layer_caches=None):
        """
        Process input through GPT decoder with optional KV-cache.
        
        Args:
            text (Tensor): Token indices of shape (batch, seq_len)
            layer_caches (list, optional): List of (past_key, past_value) tuples
                                          for each layer. None for fresh computation.
        
        Returns:
            x (Tensor): Decoder output of shape (batch, seq_len, d_model)
            presents (list): List of (key, value) tuples for caching
        """
        # Embed tokens
        x = self.embedding(text)
        
        # Calculate position offset for incremental decoding
        # When using cache, new tokens start from position = cache_length
        offset = layer_caches[0][0].size(2) if layer_caches is not None else 0
        
        # Add positional encoding (with offset for cached positions)
        x = self.pos_encoder(x, offset=offset)
        
        # Create causal mask only for fresh computation (not needed with cache)
        # With cache, attention is computed over full history automatically
        mask = self._create_causal_mask(x.size(1), x.device) if layer_caches is None else None
        
        # Process through each layer, collecting new cache entries
        presents = []
        for i, layer in enumerate(self.decoders):
            # Get layer-specific cache (or None)
            layer_cache = layer_caches[i] if layer_caches is not None else None
            x, present = layer(x, mask=mask, layer_cache=layer_cache)
            presents.append(present)
        
        return x, presents