"""LayerNorm parameter expansion.

Expands LayerNorm gamma (weight) and beta (bias) parameters to larger dimensions
while preserving function on the original subspace.

For gamma: copy original values, pad remaining with 1.0
For beta: copy original values, pad remaining with 0.0

This ensures new channels are neutral (identity scaling, zero shift) and
preserves exact behavior on inputs restricted to the original dimensions.
"""

import torch


def expand_layernorm_gamma(gamma: torch.Tensor, tgt_dim: int) -> torch.Tensor:
    """Expand LayerNorm weight (gamma) to target dimension.
    
    Copies original values into first entries, pads remaining with 1.0.
    This preserves scaling on original dimensions and applies identity
    scaling (1.0) to new dimensions.
    
    Args:
        gamma: Original LayerNorm weight tensor of shape (src_dim,).
        tgt_dim: Target dimension (must be >= src_dim).
        
    Returns:
        Expanded gamma tensor of shape (tgt_dim,).
        
    Raises:
        ValueError: If tgt_dim < src_dim (shrinking not supported).
    """
    src_dim = gamma.shape[0]
    
    if tgt_dim < src_dim:
        raise ValueError(
            f"Cannot shrink LayerNorm gamma from {src_dim} to {tgt_dim}. "
            "Target dimension must be >= source dimension."
        )
    
    if tgt_dim == src_dim:
        return gamma.clone()
    
    # Create expanded tensor initialized with 1.0
    expanded = torch.ones(tgt_dim, dtype=gamma.dtype, device=gamma.device)
    
    # Copy original values into first src_dim entries
    expanded[:src_dim] = gamma
    
    return expanded


def expand_layernorm_beta(beta: torch.Tensor, tgt_dim: int) -> torch.Tensor:
    """Expand LayerNorm bias (beta) to target dimension.
    
    Copies original values into first entries, pads remaining with 0.0.
    This preserves shift on original dimensions and applies zero shift
    to new dimensions.
    
    Args:
        beta: Original LayerNorm bias tensor of shape (src_dim,).
        tgt_dim: Target dimension (must be >= src_dim).
        
    Returns:
        Expanded beta tensor of shape (tgt_dim,).
        
    Raises:
        ValueError: If tgt_dim < src_dim (shrinking not supported).
    """
    src_dim = beta.shape[0]
    
    if tgt_dim < src_dim:
        raise ValueError(
            f"Cannot shrink LayerNorm beta from {src_dim} to {tgt_dim}. "
            "Target dimension must be >= source dimension."
        )
    
    if tgt_dim == src_dim:
        return beta.clone()
    
    # Create expanded tensor initialized with 0.0
    expanded = torch.zeros(tgt_dim, dtype=beta.dtype, device=beta.device)
    
    # Copy original values into first src_dim entries
    expanded[:src_dim] = beta
    
    return expanded


def expand_layernorm_params(
    weight: torch.Tensor,
    bias: torch.Tensor,
    tgt_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand both LayerNorm weight and bias to target dimension.
    
    Convenience function to expand both parameters together.
    
    Args:
        weight: Original LayerNorm weight (gamma) of shape (src_dim,).
        bias: Original LayerNorm bias (beta) of shape (src_dim,).
        tgt_dim: Target dimension.
        
    Returns:
        Tuple of (expanded_weight, expanded_bias).
    """
    return (
        expand_layernorm_gamma(weight, tgt_dim),
        expand_layernorm_beta(bias, tgt_dim),
    )

