"""
Attention Mechanisms for Transformer Models.

This module implements the core attention mechanisms used in transformer architectures:
- Scaled Dot-Product Attention
- Multi-Head Attention with KV-Cache support for efficient inference

References:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - https://arxiv.org/abs/1706.03762
"""

import torch
import torch.nn as nn


class DotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    Computes attention weights using the formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    The scaling factor (1/sqrt(d_k)) prevents the dot products from growing
    too large in magnitude, which would push the softmax into regions with
    extremely small gradients.
    
    Args:
        dropout (float): Dropout probability applied to attention weights. Default: 0.1
    """
    
    def __init__(self, dropout=0.1):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, keys, values, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            queries (Tensor): Query tensor of shape (batch, heads, seq_len, d_k)
            keys (Tensor): Key tensor of shape (batch, heads, seq_len, d_k)
            values (Tensor): Value tensor of shape (batch, heads, seq_len, d_k)
            mask (Tensor, optional): Attention mask. Positions with False/0 are masked out.
        
        Returns:
            Tensor: Attention output of shape (batch, heads, seq_len, d_k)
        """
        d_k = queries.size(-1)
        
        # Scale factor to prevent vanishing gradients in softmax
        denum = torch.sqrt(torch.tensor(d_k, dtype=queries.dtype, device=queries.device))
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = queries @ keys.permute(0, 1, 3, 2) / denum
        
        # Apply mask (for causal attention or padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Convert scores to probabilities
        attention = torch.softmax(scores, dim=-1)
        
        # Apply dropout and compute weighted sum of values
        output = self.dropout(attention) @ values
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism with KV-Cache support.
    
    Instead of performing a single attention function, Multi-Head Attention
    projects Q, K, V into multiple subspaces (heads), performs attention in
    parallel, then concatenates and projects the results.
    
    This implementation includes KV-Cache support for efficient autoregressive
    generation, allowing incremental decoding without recomputing past keys/values.
    
    Args:
        d_model (int): Model dimension (must be divisible by num_heads)
        num_heads (int): Number of attention heads. Default: 8
        dropout (float): Dropout probability. Default: 0.1
    
    Attributes:
        w_q, w_k, w_v (nn.Linear): Projection layers for Q, K, V
        w_o (nn.Linear): Output projection layer
        d_k (int): Dimension per head (d_model // num_heads)
    """
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model is not divisible by num_heads: {d_model} // {num_heads} = {d_model // num_heads}")
        
        # Linear projections for Query, Key, Value, and Output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_model // num_heads  # Dimension per head
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        
    def _split_into_heads(self, q, k, v):
        """
        Split tensors into multiple attention heads.
        
        Reshapes from (batch, seq_len, d_model) to (batch, num_heads, seq_len, d_k)
        
        Args:
            q, k, v (Tensor): Input tensors of shape (batch, seq_len, d_model)
        
        Returns:
            Tuple of tensors, each with shape (batch, num_heads, seq_len, d_k)
        """
        batch_size, q_len, _ = q.shape
        _, k_len, _ = k.shape
        _, v_len, _ = v.shape
        
        # Reshape: (batch, seq, d_model) -> (batch, seq, d_k, heads) -> (batch, heads, seq, d_k)
        q = q.view(batch_size, q_len, self.d_k, self.num_heads)
        q = q.permute(0, 3, 1, 2)
        
        k = k.view(batch_size, k_len, self.d_k, self.num_heads)
        k = k.permute(0, 3, 1, 2)
        
        v = v.view(batch_size, v_len, self.d_k, self.num_heads)
        v = v.permute(0, 3, 1, 2)
        
        return q, k, v

    def _concat_heads(self, weighted_output):
        """
        Concatenate attention heads back into a single tensor.
        
        Reshapes from (batch, num_heads, seq_len, d_k) to (batch, seq_len, d_model)
        
        Args:
            weighted_output (Tensor): Output from attention of shape (batch, heads, seq, d_k)
        
        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = weighted_output.shape
        
        # Reshape: (batch, heads, seq, d_k) -> (batch, seq, heads, d_k) -> (batch, seq, d_model)
        weighted_output = weighted_output.permute(0, 2, 1, 3)
        weighted_output = weighted_output.reshape(batch_size, seq_len, d_k * num_heads)
        return weighted_output
        
    def forward(self, query, key, value, mask=None, layer_cache=None):
        """
        Compute multi-head attention with optional KV-cache.
        
        Args:
            query (Tensor): Query tensor of shape (batch, seq_len, d_model)
            key (Tensor): Key tensor of shape (batch, seq_len, d_model)
            value (Tensor): Value tensor of shape (batch, seq_len, d_model)
            mask (Tensor, optional): Attention mask for causal/padding masking
            layer_cache (tuple, optional): Cached (key, value) from previous steps
                                          for efficient autoregressive generation
        
        Returns:
            output (Tensor): Attention output of shape (batch, seq_len, d_model)
            present (tuple): Current (key, value) cache to be passed to next step
        """
        # Project inputs through linear layers
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        
        # Split into multiple heads
        q, k, v = self._split_into_heads(q, k, v)
        
        # KV-Cache: Append new keys/values to cached history
        # This enables O(1) per-token generation instead of O(n)
        if layer_cache is not None:
            past_k, past_v = layer_cache
            k = torch.cat((past_k, k), dim=2)  # Concatenate along sequence dimension
            v = torch.cat((past_v, v), dim=2)
            
        # Store current K, V for next generation step
        present = (k, v)
        
        # Compute attention
        output = self.attention(q, k, v, mask)
        
        # Concatenate heads and project output
        result = self._concat_heads(output)
        
        return self.w_o(result), present
