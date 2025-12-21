"""
Transformer Models Package.

A from-scratch implementation of Transformer architectures in PyTorch,
including both encoder-decoder and decoder-only (GPT-style) variants.

Modules:
    - attention: Multi-Head Attention with KV-cache support
    - layers: Core building blocks (embeddings, FFN, normalization)
    - encoder: Transformer encoder stack
    - decoder: Transformer decoder stacks (Seq2Seq and GPT-style)
    - transformer: Complete model architectures

Example:
    >>> from models.transformer import GPTTransformer
    >>> model = GPTTransformer(vocab_size=32000, num_layers=12, d_model=768)
    >>> logits, cache = model(input_ids)
"""

from .transformer import Transformer, GPTTransformer
from .encoder import Encoder
from .decoder import Decoder, GPTDecoder

__all__ = ['Transformer', 'GPTTransformer', 'Encoder', 'Decoder', 'GPTDecoder']