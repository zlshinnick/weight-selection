"""Depth expansion for ViT models.

Maps source model blocks to target model blocks using the "early-middle-late"
scheme for procedurally pretrained models with 3 learned blocks.

Procedural structure (source):
- Layer 0: early adapter (unique)
- Layers 1..L-2: shared middle block (tied)
- Layer L-1: late refinement block (unique)

Depth expansion rule:
- target layer 0 <- source early block (layer 0)
- target layers 1..D-2 <- source middle block (layer 1)
- target layer D-1 <- source late block (layer L-1)
"""

from typing import Dict


def map_source_to_target_blocks(
    src_depth: int,
    tgt_depth: int,
    scheme: str = "early-middle-late",
) -> Dict[int, int]:
    """Create mapping from target block indices to source block indices.
    
    For the "early-middle-late" scheme:
    - target[0] <- source[0]           (early block)
    - target[1..D-2] <- source[1]      (middle block, repeated)
    - target[D-1] <- source[L-1]       (late block)
    
    Args:
        src_depth: Number of blocks in source model.
        tgt_depth: Number of blocks in target model.
        scheme: Mapping scheme. Currently only "early-middle-late" is supported.
        
    Returns:
        Dictionary mapping target block index -> source block index.
        
    Raises:
        ValueError: If scheme is not supported or dimensions are invalid.
        
    Example:
        >>> map_source_to_target_blocks(12, 12)
        {0: 0, 1: 1, 2: 1, ..., 10: 1, 11: 11}
        
        >>> map_source_to_target_blocks(12, 24)
        {0: 0, 1: 1, 2: 1, ..., 22: 1, 23: 11}
    """
    if scheme != "early-middle-late":
        raise ValueError(
            f"Unsupported depth expansion scheme: {scheme}. "
            "Only 'early-middle-late' is currently supported."
        )
    
    if src_depth < 3:
        raise ValueError(
            f"Source depth must be at least 3 for early-middle-late scheme, "
            f"got {src_depth}"
        )
    
    if tgt_depth < 3:
        raise ValueError(
            f"Target depth must be at least 3 for early-middle-late scheme, "
            f"got {tgt_depth}"
        )
    
    # Source block indices
    early_idx = 0
    middle_idx = 1  # The tied middle block
    late_idx = src_depth - 1
    
    # Build mapping
    mapping = {}
    
    # Target layer 0 <- source early block
    mapping[0] = early_idx
    
    # Target layers 1 to D-2 <- source middle block
    for tgt_idx in range(1, tgt_depth - 1):
        mapping[tgt_idx] = middle_idx
    
    # Target layer D-1 <- source late block
    mapping[tgt_depth - 1] = late_idx
    
    return mapping


def get_source_block_for_target(
    tgt_block_idx: int,
    src_depth: int,
    tgt_depth: int,
    scheme: str = "early-middle-late",
) -> int:
    """Get the source block index for a given target block index.
    
    Convenience function for single lookups without building full mapping.
    
    Args:
        tgt_block_idx: Target block index.
        src_depth: Number of blocks in source model.
        tgt_depth: Number of blocks in target model.
        scheme: Mapping scheme.
        
    Returns:
        Source block index.
    """
    if scheme != "early-middle-late":
        raise ValueError(f"Unsupported scheme: {scheme}")
    
    if tgt_block_idx < 0 or tgt_block_idx >= tgt_depth:
        raise ValueError(
            f"Target block index {tgt_block_idx} out of range [0, {tgt_depth})"
        )
    
    if tgt_block_idx == 0:
        return 0  # Early block
    elif tgt_block_idx == tgt_depth - 1:
        return src_depth - 1  # Late block
    else:
        return 1  # Middle block


def describe_depth_mapping(
    src_depth: int,
    tgt_depth: int,
    scheme: str = "early-middle-late",
) -> str:
    """Generate a human-readable description of the depth mapping.
    
    Args:
        src_depth: Number of blocks in source model.
        tgt_depth: Number of blocks in target model.
        scheme: Mapping scheme.
        
    Returns:
        Multi-line string describing the mapping.
    """
    mapping = map_source_to_target_blocks(src_depth, tgt_depth, scheme)
    
    lines = [
        f"Depth expansion: {src_depth} blocks -> {tgt_depth} blocks",
        f"Scheme: {scheme}",
        "",
        "Mapping:",
        f"  Target layer 0 <- Source layer 0 (early block)",
    ]
    
    # Count middle layers
    middle_targets = [i for i in range(1, tgt_depth - 1)]
    if middle_targets:
        if len(middle_targets) == 1:
            lines.append(f"  Target layer 1 <- Source layer 1 (middle block)")
        else:
            lines.append(
                f"  Target layers 1-{tgt_depth - 2} <- Source layer 1 "
                f"(middle block, repeated {len(middle_targets)} times)"
            )
    
    lines.append(f"  Target layer {tgt_depth - 1} <- Source layer {src_depth - 1} (late block)")
    
    return "\n".join(lines)

