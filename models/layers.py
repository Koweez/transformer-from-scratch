"""
Core Building Blocks for Transformer Models.

This module implements fundamental components used across transformer architectures:
- Positional Encoding (sinusoidal)
- Rotary Position Embedding (RoPE)
- Attention with Linear Biases (ALiBi)
- Token Embeddings with scaling
- Feed-Forward Networks (standard and GPT-style with SwiGLU)
- Normalization layers (LayerNorm and RMSNorm)

References:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
    - "Train Short, Test Long: Attention with Linear Biases" (Press et al., 2022)
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


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Encodes position information by rotating pairs of dimensions in the
    query and key vectors. This allows relative position information to
    be naturally incorporated into attention scores.
    
    Key properties:
        - Relative position is encoded through rotation
        - Decays naturally with distance (via dot product properties)
        - Can extrapolate to longer sequences than seen during training
    
    References:
        - "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
        - https://arxiv.org/abs/2104.09864
    
    Args:
        d_model (int): Model dimension (must be even)
        max_len (int): Maximum sequence length. Default: 10000
        base (float): Base for the frequency computation. Default: 10000.0
    """
    
    def __init__(self, d_model, max_len=10000, base=10000.0):
        super(RotaryPositionalEmbedding, self).__init__()
        
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # Compute inverse frequencies: theta_i = 1 / (base^(2i/d_model))
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute sin/cos for all positions up to max_len
        self._precompute_cache(max_len)
    
    def _precompute_cache(self, seq_len):
        """Precompute sin and cos values for positions."""
        # Position indices: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        
        # Outer product: [seq_len, d_model/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Duplicate frequencies for pairing: [seq_len, d_model]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Cache sin and cos: [1, seq_len, 1, d_model]
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(2))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(2))
    
    def _rotate_half(self, x):
        """
        Rotate half the hidden dims of the input.
        
        Splits x into two halves and returns [-x2, x1] for rotation.
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q, k, offset=0):
        """
        Apply rotary embeddings to queries and keys.
        
        Args:
            q (Tensor): Query tensor of shape (batch, seq_len, n_heads, head_dim)
                        or (batch, n_heads, seq_len, head_dim)
            k (Tensor): Key tensor of same shape as q
            offset (int): Position offset for KV-cache inference. Default: 0
        
        Returns:
            Tuple[Tensor, Tensor]: Rotated queries and keys
        """
        seq_len = q.shape[1] if q.dim() == 4 else q.shape[2]
        
        # Extend cache if needed
        if offset + seq_len > self.cos_cached.shape[1]:
            self._precompute_cache(offset + seq_len)
        
        # Get relevant positions
        cos = self.cos_cached[:, offset:offset + seq_len, :, :]
        sin = self.sin_cached[:, offset:offset + seq_len, :, :]
        
        # Apply rotation: x' = x * cos + rotate_half(x) * sin
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed


class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi).
    
    Instead of adding positional embeddings to token representations,
    ALiBi adds a linear bias to attention scores based on the distance
    between query and key positions. This provides:
        - No learned parameters for positions
        - Better length extrapolation than sinusoidal or learned embeddings
        - Simpler implementation
    
    The bias for head m at positions (i, j) is:
        bias(i, j) = -m * |i - j|
    
    where m is a head-specific slope that decreases geometrically.
    
    References:
        - "Train Short, Test Long: Attention with Linear Biases Enables 
           Input Length Extrapolation" (Press et al., 2022)
        - https://arxiv.org/abs/2108.12409
    
    Args:
        n_heads (int): Number of attention heads
        max_len (int): Maximum sequence length. Default: 2048
    """
    
    def __init__(self, n_heads, max_len=2048):
        super(ALiBiPositionalBias, self).__init__()
        
        self.n_heads = n_heads
        self.max_len = max_len
        
        # Compute head-specific slopes
        slopes = self._get_slopes(n_heads)
        self.register_buffer('slopes', slopes)
        
        # Precompute bias matrix for efficiency
        self._precompute_bias(max_len)
    
    def _get_slopes(self, n_heads):
        """
        Compute ALiBi slopes for each attention head.
        
        For n heads, slopes are: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
        This geometric sequence ensures different heads attend to
        different position ranges.
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(torch.log2(torch.tensor(n)).floor() - 3)))
            ratio = start # we define this way to match the original paper's formula
            return torch.tensor([start * (ratio ** i) for i in range(n)])
        
        if n_heads & (n_heads - 1) == 0:  # Power of 2
            slopes = get_slopes_power_of_2(n_heads)
        else:
            # For non-power-of-2 heads, interpolate
            closest_power_of_2 = 2 ** torch.tensor(n_heads).float().log2().floor().int() # formula: 2**floor(log2(n)) = largest power of 2 <= n
            slopes = torch.cat([
                get_slopes_power_of_2(closest_power_of_2),
                self._get_slopes(2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2]
            ])
        
        return slopes.view(1, n_heads, 1, 1)
    
    def _precompute_bias(self, max_len):
        """Precompute the relative position bias matrix."""
        # Create relative position matrix: [max_len, max_len]
        # positions[i, j] = j - i (negative for causal, key position - query position)
        positions = torch.arange(max_len).unsqueeze(0) - torch.arange(max_len).unsqueeze(1)
        
        # ALiBi uses negative distances (penalizes distant positions)
        # Shape: [1, 1, max_len, max_len]
        relative_positions = positions.unsqueeze(0).unsqueeze(0).float()
        
        self.register_buffer('relative_positions', relative_positions)
    
    def forward(self, seq_len, offset=0):
        """
        Generate ALiBi bias to add to attention scores.
        
        Args:
            seq_len (int): Current sequence length
            offset (int): Position offset for KV-cache inference. Default: 0
                         Used when generating tokens incrementally.
        
        Returns:
            Tensor: Bias tensor of shape (1, n_heads, seq_len, seq_len + offset)
                   to be added to attention scores before softmax
        """
        # Extend cache if needed
        if offset + seq_len > self.max_len:
            self._precompute_bias(offset + seq_len)
            self.max_len = offset + seq_len
        
        # Extract relevant portion of bias matrix
        # For causal attention with KV-cache:
        # - Query positions: [offset, offset + seq_len)
        # - Key positions: [0, offset + seq_len)
        positions = self.relative_positions[:, :, offset:offset + seq_len, :offset + seq_len]
        
        # Apply head-specific slopes: bias = slope * relative_position
        # Negative slopes ensure distant positions get negative bias
        alibi_bias = self.slopes * positions
        
        return alibi_bias
    
    def get_causal_bias(self, seq_len, offset=0):
        """
        Get ALiBi bias combined with causal mask.
        
        Convenience method that returns ALiBi bias with -inf for
        future positions (causal masking).
        
        Args:
            seq_len (int): Current sequence length
            offset (int): Position offset for KV-cache. Default: 0
        
        Returns:
            Tensor: Combined ALiBi + causal bias
        """
        alibi_bias = self.forward(seq_len, offset)
        
        # Create causal mask (upper triangular = -inf)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len + offset, device=alibi_bias.device) * float('-inf'),
            diagonal=1 + offset
        )
        
        return alibi_bias + causal_mask.unsqueeze(0).unsqueeze(0)


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
        self.activation = nn.SiLU()

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