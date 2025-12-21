"""
Core Building Blocks for Transformer Models.

This module implements fundamental components used across transformer architectures:
- Positional Encoding (sinusoidal)
- Token Embeddings with scaling
- Feed-Forward Networks (standard and GPT-style with SwiGLU)
- Normalization layers (LayerNorm and RMSNorm)

References:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - "GLU Variants Improve Transformer" (Shazeer, 2020)
    - "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    
    Injects position information into the model using sine and cosine functions
    of different frequencies. This allows the model to learn relative positions
    since PE(pos+k) can be expressed as a linear function of PE(pos).
    
    The encoding for position `pos` and dimension `i` is:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model (int): Model dimension
        max_len (int): Maximum sequence length. Default: 10000
        dropout (float): Dropout probability. Default: 0.1
    """
    
    def __init__(self, d_model, max_len=10000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # Position indices: [0, 1, 2, ..., max_len-1] reshaped to [max_len, 1]
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # Compute division term in exponential form for numerical stability
        # div_term = 1 / (10000^(2i/d_model)) = exp(-2i * log(10000) / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe)
        
    def forward(self, pos, offset=0):
        """
        Add positional encoding to input embeddings.
        
        Args:
            pos (Tensor): Input embeddings of shape (batch, seq_len, d_model)
            offset (int): Position offset for KV-cache inference. Default: 0
                         When using cached keys/values, offset indicates the
                         starting position for new tokens.
        
        Returns:
            Tensor: Input with positional encoding added
        """
        # Select appropriate positions (handles offset for incremental decoding)
        res = self.pe[:, offset:offset + pos.size(1), :]
        return pos + self.dropout(res)


class TokenEmbedding(nn.Module):
    """
    Token Embedding with scaling.
    
    Converts token indices to dense vectors and scales by sqrt(d_model)
    as recommended in "Attention Is All You Need" to maintain variance
    when combined with positional encodings.
    
    Args:
        vocab_size (int): Size of the vocabulary
        d_model (int): Embedding dimension
    """
    
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        Embed tokens and scale.
        
        Args:
            x (Tensor): Token indices of shape (batch, seq_len)
        
        Returns:
            Tensor: Scaled embeddings of shape (batch, seq_len, d_model)
        """
        # Scale embeddings by sqrt(d_model) to balance with positional encoding
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
    

class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function.
    
    A gated linear unit variant using SiLU (Swish) activation, shown to improve
    transformer performance over standard ReLU-based FFNs.
    
    Computes: SwiGLU(x) = SiLU(W_gate * x) ⊙ (W_up * x), then projects down.
    
    References:
        - "GLU Variants Improve Transformer" (Shazeer, 2020)
        - https://arxiv.org/abs/2002.05202
    
    Args:
        in_dim (int): Input dimension
        hidden_dim (int): Hidden/intermediate dimension
        out_dim (int): Output dimension
    """
    
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SwiGLU, self).__init__()
        self.gate = nn.Linear(in_dim, hidden_dim)  # Gate projection
        self.up = nn.Linear(in_dim, hidden_dim)    # Up projection
        self.down = nn.Linear(hidden_dim, out_dim) # Down projection
        self.silu = nn.SiLU()  # Swish activation: x * sigmoid(x)
        
    def forward(self, x):
        """
        Apply SwiGLU transformation.
        
        Args:
            x (Tensor): Input tensor
        
        Returns:
            Tensor: Transformed tensor
        """
        gate = self.gate(x)
        up = self.up(x)
        x = self.silu(gate) * up  # Gated activation
        x = self.down(x)
        return x


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (Standard Transformer).
    
    Two-layer MLP applied independently to each position:
        FFN(x) = Linear2(Dropout(Activation(Linear1(x))))
    
    This implementation uses SwiGLU activation for improved performance.
    
    Args:
        d_model (int): Model dimension (input and output)
        d_ff (int): Inner/hidden dimension (typically 4x d_model)
        dropout (float): Dropout probability. Default: 0.1
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = SwiGLU(d_model, d_ff, d_ff)

    def forward(self, x):
        """
        Apply feed-forward transformation.
        
        Args:
            x (Tensor): Input of shape (batch, seq_len, d_model)
        
        Returns:
            Tensor: Output of shape (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class GPTFeedForward(nn.Module):
    """
    GPT-style Feed-Forward Network with SwiGLU.
    
    Simplified FFN using SwiGLU activation, commonly used in modern LLMs
    like LLaMA and Mistral. More parameter-efficient than standard FFN.
    
    Args:
        d_model (int): Model dimension
        d_ff (int): Hidden dimension for SwiGLU
        dropout (float): Dropout probability. Default: 0.1
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(GPTFeedForward, self).__init__()
        self.swiglu = SwiGLU(d_model, d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Apply GPT-style feed-forward transformation.
        
        Args:
            x (Tensor): Input of shape (batch, seq_len, d_model)
        
        Returns:
            Tensor: Output of shape (batch, seq_len, d_model)
        """
        x = self.swiglu(x)
        x = self.dropout(x)
        return x
    

class LayerNorm(nn.Module):
    """
    Layer Normalization.
    
    Normalizes inputs across the feature dimension with learnable
    scale (gamma) and shift (beta) parameters.
    
    LayerNorm(x) = gamma * (x - mean) / (std + eps) + beta
    
    Args:
        d_model (int): Feature dimension to normalize over
        eps (float): Small constant for numerical stability. Default: 1e-6
    """
    
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))   # Learnable scale
        self.beta = nn.Parameter(torch.zeros(d_model))   # Learnable shift
        self.eps = eps

    def forward(self, x):
        """
        Apply layer normalization.
        
        Args:
            x (Tensor): Input tensor with last dim = d_model
        
        Returns:
            Tensor: Normalized tensor
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    A simplified normalization that only rescales by the RMS (no mean centering),
    shown to be as effective as LayerNorm while being computationally cheaper.
    Used in modern LLMs like LLaMA.
    
    RMSNorm(x) = x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)
    
    References:
        - "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
        - https://arxiv.org/abs/1910.07467
    
    Args:
        d_model (int): Feature dimension
        eps (float): Small constant for numerical stability. Default: 1e-8
    """
    
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # Learnable scale
        self.eps = eps

    def forward(self, x):
        """
        Apply RMS normalization.
        
        Args:
            x (Tensor): Input tensor with last dim = d_model
        
        Returns:
            Tensor: Normalized tensor
        """
        # rsqrt = 1/sqrt for efficiency
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.gamma