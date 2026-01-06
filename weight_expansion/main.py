"""Main API for ViT weight expansion.

Expands procedurally pretrained ViT transformer block weights to larger models
using SVD + row-wise tiling for width expansion and early-middle-late block
mapping for depth expansion.

Function-preserving: for inputs restricted to the original hidden dimensions
(zero-padded into the expanded model), the expanded weights reproduce the
original computation on those dimensions at initialization.
"""

from typing import Dict, Any
from pathlib import Path

import torch

from .utils import (
    load_checkpoint,
    parse_block_key,
    filter_block_params,
    get_block_indices,
    make_block_key,
    validate_config,
)
from .depth_expand import (
    map_source_to_target_blocks,
    describe_depth_mapping,
)
from .width_expand import (
    expand_qkv_weight,
    expand_fc1_weight,
    expand_fc2_weight,
    expand_proj_weight,
)
from .layernorm_expand import (
    expand_layernorm_gamma,
    expand_layernorm_beta,
)
from .bias_expand import (
    expand_bias,
    expand_qkv_bias,
)
from .validation import (
    assert_shapes,
    verify_full_expansion,
)
from .gpt2_convert import (
    gpt2_to_vit_state_dict,
    vit_to_gpt2_state_dict,
)


# Required config keys
REQUIRED_CONFIG_KEYS = ["embed_dim", "depth"]


def _expand_block_params(
    src_block_sd: Dict[str, torch.Tensor],
    src_cfg: Dict[str, Any],
    tgt_cfg: Dict[str, Any],
    tile_mode: str,
    scale_mode: str,
    pad_mode: str,
) -> Dict[str, torch.Tensor]:
    """Expand all parameters for a single block.

    Args:
        src_block_sd: Source block state dict (keys without block prefix).
        src_cfg: Source configuration.
        tgt_cfg: Target configuration.
        tile_mode: Tiling mode for SVD expansion (only used when pad_mode="tile").
        scale_mode: Scaling mode for SVD expansion.
        pad_mode: Expansion strategy: "tile", "zeros", or "random".

    Returns:
        Expanded block state dict (keys without block prefix).
    """
    src_embed_dim = src_cfg["embed_dim"]
    tgt_embed_dim = tgt_cfg["embed_dim"]
    src_mlp_ratio = src_cfg.get("mlp_ratio", 4)
    tgt_mlp_ratio = tgt_cfg.get("mlp_ratio", 4)
    src_mlp_dim = int(src_embed_dim * src_mlp_ratio)
    tgt_mlp_dim = int(tgt_embed_dim * tgt_mlp_ratio)

    expanded = {}

    for component, tensor in src_block_sd.items():
        # Determine parameter type and expand accordingly
        if component == "attn.qkv.weight":
            expanded[component] = expand_qkv_weight(
                tensor, src_embed_dim, tgt_embed_dim, tile_mode, scale_mode, pad_mode
            )
        elif component == "attn.qkv.bias":
            expanded[component] = expand_qkv_bias(tensor, src_embed_dim, tgt_embed_dim)
        elif component == "attn.proj.weight":
            expanded[component] = expand_proj_weight(
                tensor, src_embed_dim, tgt_embed_dim, tile_mode, scale_mode, pad_mode
            )
        elif component == "attn.proj.bias":
            expanded[component] = expand_bias(tensor, tgt_embed_dim)
        elif component == "mlp.fc1.weight":
            expanded[component] = expand_fc1_weight(
                tensor,
                src_embed_dim,
                tgt_embed_dim,
                src_mlp_dim,
                tgt_mlp_dim,
                tile_mode,
                scale_mode,
                pad_mode,
            )
        elif component == "mlp.fc1.bias":
            expanded[component] = expand_bias(tensor, tgt_mlp_dim)
        elif component == "mlp.fc2.weight":
            expanded[component] = expand_fc2_weight(
                tensor,
                src_embed_dim,
                tgt_embed_dim,
                src_mlp_dim,
                tgt_mlp_dim,
                tile_mode,
                scale_mode,
                pad_mode,
            )
        elif component == "mlp.fc2.bias":
            expanded[component] = expand_bias(tensor, tgt_embed_dim)
        elif component in ("norm1.weight", "norm2.weight"):
            expanded[component] = expand_layernorm_gamma(tensor, tgt_embed_dim)
        elif component in ("norm1.bias", "norm2.bias"):
            expanded[component] = expand_layernorm_beta(tensor, tgt_embed_dim)
        else:
            # Unknown component, copy as-is with warning
            print(f"Warning: Unknown component '{component}', copying as-is")
            expanded[component] = tensor.clone()

    return expanded


def expand_vit_state_dict(
    src_sd: Dict[str, torch.Tensor],
    src_cfg: Dict[str, Any],
    tgt_cfg: Dict[str, Any],
    depth_scheme: str = "early-middle-late",
    width_scheme: str = "svd_row_tile",
    tile_mode: str = "cyclic",
    scale_mode: str = "none",
    pad_mode: str = "tile",
    validate: bool = True,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Expand ViT state dict from source to target configuration.

    Handles only transformer block parameters:
    - Attention (qkv, proj)
    - MLP (fc1, fc2)
    - LayerNorm (gamma and beta)

    Ignores patch embedding, positional embeddings, and class token.

    Args:
        src_sd: Source state dict (can include non-block params, they'll be filtered).
        src_cfg: Source config with keys: embed_dim, depth, mlp_ratio (optional).
        tgt_cfg: Target config with same keys.
        depth_scheme: "early-middle-late" (only supported scheme).
        width_scheme: "svd_row_tile" (only supported scheme).
        tile_mode: "cyclic" or "repeat" for row tiling (only used when pad_mode="tile").
        scale_mode: "fro" (rescale to match Frobenius norm) or "none".
        pad_mode: Expansion strategy: "tile", "zeros", or "random".
        validate: If True, run shape validation after expansion.
        verbose: If True, print progress information.

    Returns:
        Expanded state dict for target model (block params and final norm only).

    Raises:
        ValueError: If unsupported scheme or invalid configuration.
        AssertionError: If validation fails.

    Example:
        >>> src_cfg = {"embed_dim": 192, "depth": 12, "mlp_ratio": 4}
        >>> tgt_cfg = {"embed_dim": 384, "depth": 12, "mlp_ratio": 4}
        >>> expanded_sd = expand_vit_state_dict(
        ...     src_sd=checkpoint["model"],
        ...     src_cfg=src_cfg,
        ...     tgt_cfg=tgt_cfg,
        ... )
    """
    # Validate inputs
    if depth_scheme != "early-middle-late":
        raise ValueError(
            f"Unsupported depth_scheme: {depth_scheme}. "
            "Only 'early-middle-late' is supported."
        )

    if width_scheme != "svd_row_tile":
        raise ValueError(
            f"Unsupported width_scheme: {width_scheme}. "
            "Only 'svd_row_tile' is supported."
        )

    validate_config(src_cfg, REQUIRED_CONFIG_KEYS)
    validate_config(tgt_cfg, REQUIRED_CONFIG_KEYS)

    src_depth = src_cfg["depth"]
    tgt_depth = tgt_cfg["depth"]
    src_embed_dim = src_cfg["embed_dim"]
    tgt_embed_dim = tgt_cfg["embed_dim"]

    if verbose:
        print("=" * 60)
        print("ViT Weight Expansion")
        print("=" * 60)
        print(f"Source: embed_dim={src_embed_dim}, depth={src_depth}")
        print(f"Target: embed_dim={tgt_embed_dim}, depth={tgt_depth}")
        print(
            f"Width scheme: {width_scheme} (tile_mode={tile_mode}, scale_mode={scale_mode}, pad_mode={pad_mode})"
        )
        print(f"Depth scheme: {depth_scheme}")
        print()

    # Filter to block parameters only
    src_block_sd = filter_block_params(src_sd)

    if verbose:
        print(f"Filtered to {len(src_block_sd)} block parameters")

    # Get source block indices
    src_block_indices = get_block_indices(src_block_sd)
    if verbose:
        print(f"Source blocks: {src_block_indices}")

    # Create depth mapping
    block_mapping = map_source_to_target_blocks(src_depth, tgt_depth, depth_scheme)

    if verbose:
        print()
        print(describe_depth_mapping(src_depth, tgt_depth, depth_scheme))
        print()

    # Organize source params by block
    src_blocks = {}
    for key, tensor in src_block_sd.items():
        parsed = parse_block_key(key)
        if parsed is not None:
            block_idx, component = parsed
            if block_idx not in src_blocks:
                src_blocks[block_idx] = {}
            src_blocks[block_idx][component] = tensor

    # Also handle final norm
    final_norm = {}
    if "norm.weight" in src_block_sd:
        final_norm["weight"] = src_block_sd["norm.weight"]
    if "norm.bias" in src_block_sd:
        final_norm["bias"] = src_block_sd["norm.bias"]

    # Expand each target block
    expanded_sd = {}

    # Cache expanded blocks to avoid redundant computation
    expanded_block_cache = {}

    for tgt_idx in range(tgt_depth):
        src_idx = block_mapping[tgt_idx]

        if verbose:
            print(
                f"Expanding target block {tgt_idx} <- source block {src_idx}...",
                end=" ",
            )

        # Use cached expansion if available
        if src_idx in expanded_block_cache:
            expanded_block = expanded_block_cache[src_idx]
            if verbose:
                print("(cached)")
        else:
            # Expand the source block
            src_block = src_blocks.get(src_idx, {})
            if not src_block:
                raise ValueError(f"Source block {src_idx} not found in state dict")

            expanded_block = _expand_block_params(
                src_block, src_cfg, tgt_cfg, tile_mode, scale_mode, pad_mode
            )
            expanded_block_cache[src_idx] = expanded_block
            if verbose:
                print("done")

        # Add to expanded state dict with target block prefix
        for component, tensor in expanded_block.items():
            key = make_block_key(tgt_idx, component)
            # Clone to ensure each block has its own copy
            expanded_sd[key] = tensor.clone()

    # Expand final norm
    if final_norm:
        if verbose:
            print("Expanding final norm...", end=" ")
        if "weight" in final_norm:
            expanded_sd["norm.weight"] = expand_layernorm_gamma(
                final_norm["weight"], tgt_embed_dim
            )
        if "bias" in final_norm:
            expanded_sd["norm.bias"] = expand_layernorm_beta(
                final_norm["bias"], tgt_embed_dim
            )
        if verbose:
            print("done")

    if verbose:
        print()
        print(f"Expanded state dict has {len(expanded_sd)} parameters")

    # Validate shapes
    if validate:
        if verbose:
            print("\nValidating shapes...")
        assert_shapes(expanded_sd, tgt_cfg, tgt_depth)
        if verbose:
            print("✓ Shape validation passed!")

    if verbose:
        print()
        print("=" * 60)
        print("Expansion complete!")
        print("=" * 60)

    return expanded_sd


def expand_and_save(
    src_path: str | Path,
    tgt_path: str | Path,
    src_cfg: Dict[str, Any],
    tgt_cfg: Dict[str, Any],
    depth_scheme: str = "early-middle-late",
    width_scheme: str = "svd_row_tile",
    tile_mode: str = "cyclic",
    scale_mode: str = "none",
    pad_mode: str = "tile",
    validate: bool = True,
    verify_function: bool = False,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Load checkpoint, expand weights, and save to new path.

    Convenience function for command-line usage.

    Args:
        src_path: Path to source checkpoint.
        tgt_path: Path for output checkpoint.
        src_cfg: Source configuration.
        tgt_cfg: Target configuration.
        depth_scheme: Depth expansion scheme.
        width_scheme: Width expansion scheme.
        tile_mode: Tiling mode (only used when pad_mode="tile").
        scale_mode: Scaling mode.
        pad_mode: Expansion strategy: "tile", "zeros", or "random".
        validate: Run shape validation.
        verify_function: Run function preservation verification.
        verbose: Print progress.

    Returns:
        Expanded state dict.
    """
    src_path = Path(src_path)
    tgt_path = Path(tgt_path)

    if verbose:
        print(f"Loading checkpoint from {src_path}...")

    src_sd, wrapper, wrapper_key = load_checkpoint(src_path)

    if verbose:
        if wrapper_key:
            print(f"Found state dict under key: '{wrapper_key}'")
        print(f"Loaded {len(src_sd)} parameters")

    # Expand
    expanded_sd = expand_vit_state_dict(
        src_sd=src_sd,
        src_cfg=src_cfg,
        tgt_cfg=tgt_cfg,
        depth_scheme=depth_scheme,
        width_scheme=width_scheme,
        tile_mode=tile_mode,
        scale_mode=scale_mode,
        pad_mode=pad_mode,
        validate=validate,
        verbose=verbose,
    )

    # Optionally verify function preservation
    if verify_function:
        if verbose:
            print("\nVerifying function preservation...")
        verify_full_expansion(
            src_sd=filter_block_params(src_sd),
            expanded_sd=expanded_sd,
            src_cfg=src_cfg,
            tgt_cfg=tgt_cfg,
            verbose=verbose,
        )

    # Save
    tgt_path.parent.mkdir(parents=True, exist_ok=True)

    # Wrap in dict for consistency
    output = {"model": expanded_sd}

    if verbose:
        print(f"\nSaving to {tgt_path}...")

    torch.save(output, tgt_path)

    if verbose:
        src_size = src_path.stat().st_size / (1024 * 1024)
        tgt_size = tgt_path.stat().st_size / (1024 * 1024)
        print(f"✓ Saved: {src_size:.1f} MB -> {tgt_size:.1f} MB")

    return expanded_sd


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Expand ViT weights from smaller to larger model"
    )
    parser.add_argument(
        "--src_checkpoint",
        type=str,
        help="Path to source checkpoint",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output checkpoint",
    )
    parser.add_argument(
        "--src-embed-dim",
        type=int,
        default=192,
        help="Source embedding dimension (default: 192 for ViT-Tiny)",
    )
    parser.add_argument(
        "--src-depth",
        type=int,
        default=12,
        help="Source model depth (default: 12)",
    )
    parser.add_argument(
        "--tgt-embed-dim",
        type=int,
        default=384,
        help="Target embedding dimension (default: 384 for ViT-Small)",
    )
    parser.add_argument(
        "--tgt-depth",
        type=int,
        default=12,
        help="Target model depth (default: 12)",
    )
    parser.add_argument(
        "--mlp-ratio",
        type=float,
        default=4.0,
        help="MLP ratio (default: 4.0)",
    )
    parser.add_argument(
        "--tile-mode",
        type=str,
        choices=["cyclic", "repeat"],
        default="cyclic",
        help="Tiling mode for SVD expansion (default: cyclic)",
    )
    parser.add_argument(
        "--scale-mode",
        type=str,
        choices=["fro", "none"],
        default="none",
        help="Scaling mode (default: fro)",
    )
    parser.add_argument(
        "--pad-mode",
        type=str,
        choices=["tile", "zeros", "random"],
        default="tile",
        help="Expansion strategy: tile (SVD + tiling), zeros (zero-pad factors), random (random-pad factors with GPT-2 init). Default: tile",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify function preservation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    src_cfg = {
        "embed_dim": args.src_embed_dim,
        "depth": args.src_depth,
        "mlp_ratio": args.mlp_ratio,
    }
    tgt_cfg = {
        "embed_dim": args.tgt_embed_dim,
        "depth": args.tgt_depth,
        "mlp_ratio": args.mlp_ratio,
    }

    expand_and_save(
        src_path=args.src_checkpoint,
        tgt_path=args.output,
        src_cfg=src_cfg,
        tgt_cfg=tgt_cfg,
        tile_mode=args.tile_mode,
        scale_mode=args.scale_mode,
        pad_mode=args.pad_mode,
        verify_function=args.verify,
        verbose=not args.quiet,
    )
