"""ViT Weight Expansion Package.

Expands procedurally pretrained ViT transformer block weights to larger models
using SVD + row-wise tiling for width expansion and early-middle-late block
mapping for depth expansion.

Main API:
    expand_vit_state_dict: Expand ViT state dict from source to target configuration
    expand_and_save: Load, expand, and save checkpoint

Example:
    >>> from weight_expansion import expand_vit_state_dict
    >>> 
    >>> src_cfg = {"embed_dim": 192, "depth": 12, "mlp_ratio": 4}
    >>> tgt_cfg = {"embed_dim": 384, "depth": 12, "mlp_ratio": 4}
    >>> 
    >>> expanded_sd = expand_vit_state_dict(
    ...     src_sd=checkpoint["model"],
    ...     src_cfg=src_cfg,
    ...     tgt_cfg=tgt_cfg,
    ... )
"""

from .main import (
    expand_vit_state_dict,
    expand_and_save,
)

from .utils import (
    load_checkpoint,
    parse_block_key,
    get_param_type,
    ParamType,
    filter_block_params,
    get_block_indices,
)

from .depth_expand import (
    map_source_to_target_blocks,
    get_source_block_for_target,
    describe_depth_mapping,
)

from .width_expand import (
    expand_weight_svd_tile,
    expand_qkv_weight,
    expand_fc1_weight,
    expand_fc2_weight,
    expand_proj_weight,
)

from .layernorm_expand import (
    expand_layernorm_gamma,
    expand_layernorm_beta,
    expand_layernorm_params,
)

from .bias_expand import (
    expand_bias,
    expand_qkv_bias,
)

from .validation import (
    assert_shapes,
    verify_function_preservation_linear,
    verify_block_function_preservation,
    verify_full_expansion,
    get_expected_shapes,
)

__all__ = [
    # Main API
    "expand_vit_state_dict",
    "expand_and_save",
    # Utils
    "load_checkpoint",
    "parse_block_key",
    "get_param_type",
    "ParamType",
    "filter_block_params",
    "get_block_indices",
    # Depth expansion
    "map_source_to_target_blocks",
    "get_source_block_for_target",
    "describe_depth_mapping",
    # Width expansion
    "expand_weight_svd_tile",
    "expand_qkv_weight",
    "expand_fc1_weight",
    "expand_fc2_weight",
    "expand_proj_weight",
    # LayerNorm expansion
    "expand_layernorm_gamma",
    "expand_layernorm_beta",
    "expand_layernorm_params",
    # Bias expansion
    "expand_bias",
    "expand_qkv_bias",
    # Validation
    "assert_shapes",
    "verify_function_preservation_linear",
    "verify_block_function_preservation",
    "verify_full_expansion",
    "get_expected_shapes",
]

