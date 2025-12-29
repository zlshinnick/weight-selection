"""Bias vector expansion.

Expands bias vectors for linear layers (attention and MLP) to larger dimensions
while preserving function on the original subspace.

Bias expansion: copy original values, pad remaining with 0.0

This ensures new output dimensions have zero bias contribution and
preserves exact behavior on the original subspace.
"""

import torch


def expand_bias(bias: torch.Tensor, tgt_dim: int) -> torch.Tensor:
    """Expand a bias vector to target dimension.
    
    Copies original values into first entries, pads remaining with 0.0.
    This preserves bias on original dimensions and applies zero bias
    to new dimensions.
    
    Args:
        bias: Original bias tensor of shape (src_dim,).
        tgt_dim: Target dimension (must be >= src_dim).
        
    Returns:
        Expanded bias tensor of shape (tgt_dim,).
        
    Raises:
        ValueError: If tgt_dim < src_dim (shrinking not supported).
    """
    src_dim = bias.shape[0]
    
    if tgt_dim < src_dim:
        raise ValueError(
            f"Cannot shrink bias from {src_dim} to {tgt_dim}. "
            "Target dimension must be >= source dimension."
        )
    
    if tgt_dim == src_dim:
        return bias.clone()
    
    # Create expanded tensor initialized with 0.0
    expanded = torch.zeros(tgt_dim, dtype=bias.dtype, device=bias.device)
    
    # Copy original values into first src_dim entries
    expanded[:src_dim] = bias
    
    return expanded


def expand_qkv_bias(
    qkv_bias: torch.Tensor,
    src_embed_dim: int,
    tgt_embed_dim: int
) -> torch.Tensor:
    """Expand QKV bias which has shape (3 * embed_dim,).
    
    The QKV bias is structured as [Q_bias, K_bias, V_bias], each of size
    embed_dim. We expand each section independently and concatenate.
    
    Args:
        qkv_bias: Original QKV bias of shape (3 * src_embed_dim,).
        src_embed_dim: Source embedding dimension.
        tgt_embed_dim: Target embedding dimension.
        
    Returns:
        Expanded QKV bias of shape (3 * tgt_embed_dim,).
    """
    # Split into Q, K, V sections
    q_bias = qkv_bias[:src_embed_dim]
    k_bias = qkv_bias[src_embed_dim:2*src_embed_dim]
    v_bias = qkv_bias[2*src_embed_dim:]
    
    # Expand each section
    q_expanded = expand_bias(q_bias, tgt_embed_dim)
    k_expanded = expand_bias(k_bias, tgt_embed_dim)
    v_expanded = expand_bias(v_bias, tgt_embed_dim)
    
    # Concatenate
    return torch.cat([q_expanded, k_expanded, v_expanded], dim=0)

