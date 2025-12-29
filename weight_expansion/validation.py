"""Validation utilities for weight expansion.

Provides shape assertions and function-preservation verification
to ensure expanded weights are correct.
"""

from typing import Dict, Any, Optional
import torch

from .utils import (
    parse_block_key,
    get_param_type,
    ParamType,
)


def get_expected_shapes(
    tgt_cfg: Dict[str, Any],
    tgt_depth: int,
) -> Dict[str, tuple]:
    """Generate expected shapes for all parameters in target model.
    
    Args:
        tgt_cfg: Target configuration with embed_dim, mlp_ratio.
        tgt_depth: Target model depth (number of blocks).
        
    Returns:
        Dictionary mapping parameter keys to expected shapes.
    """
    embed_dim = tgt_cfg["embed_dim"]
    mlp_ratio = tgt_cfg.get("mlp_ratio", 4)
    mlp_dim = int(embed_dim * mlp_ratio)
    
    expected = {}
    
    # Block parameters
    for block_idx in range(tgt_depth):
        prefix = f"blocks.{block_idx}"
        
        # LayerNorm
        expected[f"{prefix}.norm1.weight"] = (embed_dim,)
        expected[f"{prefix}.norm1.bias"] = (embed_dim,)
        expected[f"{prefix}.norm2.weight"] = (embed_dim,)
        expected[f"{prefix}.norm2.bias"] = (embed_dim,)
        
        # Attention
        expected[f"{prefix}.attn.qkv.weight"] = (3 * embed_dim, embed_dim)
        expected[f"{prefix}.attn.qkv.bias"] = (3 * embed_dim,)
        expected[f"{prefix}.attn.proj.weight"] = (embed_dim, embed_dim)
        expected[f"{prefix}.attn.proj.bias"] = (embed_dim,)
        
        # MLP
        expected[f"{prefix}.mlp.fc1.weight"] = (mlp_dim, embed_dim)
        expected[f"{prefix}.mlp.fc1.bias"] = (mlp_dim,)
        expected[f"{prefix}.mlp.fc2.weight"] = (embed_dim, mlp_dim)
        expected[f"{prefix}.mlp.fc2.bias"] = (embed_dim,)
    
    # Final norm
    expected["norm.weight"] = (embed_dim,)
    expected["norm.bias"] = (embed_dim,)
    
    return expected


def assert_shapes(
    expanded_sd: Dict[str, torch.Tensor],
    tgt_cfg: Dict[str, Any],
    tgt_depth: int,
) -> None:
    """Verify all expanded tensors have expected target shapes.
    
    Args:
        expanded_sd: Expanded state dictionary.
        tgt_cfg: Target configuration.
        tgt_depth: Target model depth.
        
    Raises:
        AssertionError: If any tensor has unexpected shape.
    """
    expected = get_expected_shapes(tgt_cfg, tgt_depth)
    
    errors = []
    
    for key, tensor in expanded_sd.items():
        if key not in expected:
            # Skip keys we don't have expectations for (e.g., embeddings)
            continue
        
        expected_shape = expected[key]
        actual_shape = tuple(tensor.shape)
        
        if actual_shape != expected_shape:
            errors.append(
                f"  {key}: expected {expected_shape}, got {actual_shape}"
            )
    
    # Check for missing keys
    for key in expected:
        if key not in expanded_sd:
            errors.append(f"  {key}: MISSING")
    
    if errors:
        error_msg = "Shape validation failed:\n" + "\n".join(errors)
        raise AssertionError(error_msg)


def verify_function_preservation_linear(
    W_orig: torch.Tensor,
    W_expanded: torch.Tensor,
    n_samples: int = 100,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> tuple[bool, float]:
    """Verify function preservation for a linear layer.
    
    Tests that for inputs restricted to the original dimensions (zero-padded
    into the expanded model), the expanded weights reproduce the original
    computation on those dimensions.
    
    W_expanded([x, 0])[:d] ≈ W_orig(x)
    
    Args:
        W_orig: Original weight matrix of shape (out_dim, in_dim).
        W_expanded: Expanded weight matrix of shape (tgt_out_dim, tgt_in_dim).
        n_samples: Number of random samples to test.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
        
    Returns:
        Tuple of (passed, max_relative_error).
    """
    out_dim, in_dim = W_orig.shape
    tgt_out_dim, tgt_in_dim = W_expanded.shape
    
    # Generate random inputs in original dimension
    x = torch.randn(n_samples, in_dim, dtype=W_orig.dtype, device=W_orig.device)
    
    # Original output
    y_orig = x @ W_orig.T  # (n_samples, out_dim)
    
    # Zero-pad input to expanded dimension
    x_padded = torch.zeros(n_samples, tgt_in_dim, dtype=x.dtype, device=x.device)
    x_padded[:, :in_dim] = x
    
    # Expanded output (take first out_dim dimensions)
    y_expanded = x_padded @ W_expanded.T  # (n_samples, tgt_out_dim)
    y_expanded_truncated = y_expanded[:, :out_dim]
    
    # Compare
    max_abs_error = torch.max(torch.abs(y_orig - y_expanded_truncated)).item()
    max_orig = torch.max(torch.abs(y_orig)).item()
    
    if max_orig > 0:
        max_rel_error = max_abs_error / max_orig
    else:
        max_rel_error = max_abs_error
    
    passed = torch.allclose(y_orig, y_expanded_truncated, rtol=rtol, atol=atol)
    
    return passed, max_rel_error


def verify_block_function_preservation(
    src_sd: Dict[str, torch.Tensor],
    expanded_sd: Dict[str, torch.Tensor],
    block_idx: int,
    src_embed_dim: int,
    tgt_embed_dim: int,
    n_samples: int = 100,
    rtol: float = 1e-4,
    verbose: bool = False,
) -> Dict[str, tuple[bool, float]]:
    """Verify function preservation for all weights in a block.
    
    Args:
        src_sd: Source state dictionary.
        expanded_sd: Expanded state dictionary.
        block_idx: Block index to verify.
        src_embed_dim: Source embedding dimension.
        tgt_embed_dim: Target embedding dimension.
        n_samples: Number of random samples.
        rtol: Relative tolerance.
        verbose: Print results.
        
    Returns:
        Dictionary mapping component names to (passed, max_rel_error) tuples.
    """
    results = {}
    
    # Components to check (only 2D weight matrices)
    components = [
        ("attn.qkv.weight", "QKV"),
        ("attn.proj.weight", "Proj"),
        ("mlp.fc1.weight", "FC1"),
        ("mlp.fc2.weight", "FC2"),
    ]
    
    prefix = f"blocks.{block_idx}"
    
    for component, name in components:
        src_key = f"{prefix}.{component}"
        
        if src_key not in src_sd or src_key not in expanded_sd:
            continue
        
        W_orig = src_sd[src_key]
        W_expanded = expanded_sd[src_key]
        
        passed, max_rel_error = verify_function_preservation_linear(
            W_orig, W_expanded, n_samples, rtol
        )
        
        results[name] = (passed, max_rel_error)
        
        if verbose:
            status = "✓" if passed else "✗"
            print(f"  {status} {name}: max_rel_error = {max_rel_error:.2e}")
    
    return results


def verify_full_expansion(
    src_sd: Dict[str, torch.Tensor],
    expanded_sd: Dict[str, torch.Tensor],
    src_cfg: Dict[str, Any],
    tgt_cfg: Dict[str, Any],
    n_samples: int = 100,
    rtol: float = 1e-4,
    verbose: bool = True,
) -> bool:
    """Verify function preservation for the entire expansion.
    
    Args:
        src_sd: Source state dictionary.
        expanded_sd: Expanded state dictionary.
        src_cfg: Source configuration.
        tgt_cfg: Target configuration.
        n_samples: Number of random samples.
        rtol: Relative tolerance.
        verbose: Print results.
        
    Returns:
        True if all verifications pass.
    """
    from .utils import get_block_indices
    from .depth_expand import map_source_to_target_blocks
    
    src_embed_dim = src_cfg["embed_dim"]
    tgt_embed_dim = tgt_cfg["embed_dim"]
    src_depth = src_cfg["depth"]
    tgt_depth = tgt_cfg["depth"]
    
    # Get block mapping
    block_mapping = map_source_to_target_blocks(src_depth, tgt_depth)
    
    all_passed = True
    
    if verbose:
        print(f"Verifying function preservation...")
        print(f"  Source: {src_embed_dim}d, {src_depth} blocks")
        print(f"  Target: {tgt_embed_dim}d, {tgt_depth} blocks")
        print()
    
    # For each unique source block, verify against its corresponding expanded block
    checked_sources = set()
    
    for tgt_idx, src_idx in block_mapping.items():
        if src_idx in checked_sources:
            continue
        checked_sources.add(src_idx)
        
        if verbose:
            print(f"Block {tgt_idx} (from source block {src_idx}):")
        
        # Create temporary dicts with just this block for verification
        src_block_sd = {
            k: v for k, v in src_sd.items() 
            if k.startswith(f"blocks.{src_idx}.")
        }
        # Rename to match target block index for comparison
        src_block_renamed = {
            k.replace(f"blocks.{src_idx}.", f"blocks.{tgt_idx}."): v
            for k, v in src_block_sd.items()
        }
        
        tgt_block_sd = {
            k: v for k, v in expanded_sd.items()
            if k.startswith(f"blocks.{tgt_idx}.")
        }
        
        results = verify_block_function_preservation(
            src_block_renamed,
            tgt_block_sd,
            tgt_idx,
            src_embed_dim,
            tgt_embed_dim,
            n_samples,
            rtol,
            verbose,
        )
        
        for name, (passed, _) in results.items():
            if not passed:
                all_passed = False
        
        if verbose:
            print()
    
    if verbose:
        if all_passed:
            print("✓ All function preservation checks passed!")
        else:
            print("✗ Some function preservation checks failed.")
    
    return all_passed

