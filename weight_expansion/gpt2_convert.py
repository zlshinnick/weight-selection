"""Convert ViT state dict to HuggingFace GPT-2 format.

Handles key renaming and weight transposition for Conv1D compatibility.

HuggingFace GPT-2 uses Conv1D which stores weights as (in_features, out_features),
the transpose of PyTorch Linear (out_features, in_features).
"""

import re
from typing import Dict

import torch


# Mapping from ViT component names to GPT-2 component names
# Components marked with _TRANSPOSE need weight transposition
VIT_TO_GPT2_COMPONENT_MAP = {
    # LayerNorm (no transpose needed)
    "norm1.weight": "ln_1.weight",
    "norm1.bias": "ln_1.bias",
    "norm2.weight": "ln_2.weight",
    "norm2.bias": "ln_2.bias",
    # Attention
    "attn.qkv.weight": "attn.c_attn.weight",  # transpose
    "attn.qkv.bias": "attn.c_attn.bias",
    "attn.proj.weight": "attn.c_proj.weight",  # transpose
    "attn.proj.bias": "attn.c_proj.bias",
    # MLP
    "mlp.fc1.weight": "mlp.c_fc.weight",  # transpose
    "mlp.fc1.bias": "mlp.c_fc.bias",
    "mlp.fc2.weight": "mlp.c_proj.weight",  # transpose
    "mlp.fc2.bias": "mlp.c_proj.bias",
}

# Components that need weight transposition (Conv1D format)
TRANSPOSE_COMPONENTS = {
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
}


def _convert_block_key(vit_key: str, add_prefix: bool = True) -> tuple[str, bool]:
    """Convert a ViT block key to GPT-2 format.

    Args:
        vit_key: ViT key like "blocks.0.attn.qkv.weight"
        add_prefix: Whether to add "transformer." prefix

    Returns:
        Tuple of (gpt2_key, needs_transpose)
    """
    # Match block keys
    match = re.match(r"blocks\.(\d+)\.(.+)", vit_key)
    if not match:
        return None, False

    block_idx = match.group(1)
    component = match.group(2)

    if component not in VIT_TO_GPT2_COMPONENT_MAP:
        return None, False

    gpt2_component = VIT_TO_GPT2_COMPONENT_MAP[component]
    needs_transpose = gpt2_component in TRANSPOSE_COMPONENTS

    if add_prefix:
        gpt2_key = f"transformer.h.{block_idx}.{gpt2_component}"
    else:
        gpt2_key = f"h.{block_idx}.{gpt2_component}"

    return gpt2_key, needs_transpose


def vit_to_gpt2_state_dict(
    vit_sd: Dict[str, torch.Tensor],
    add_prefix: bool = False,
) -> Dict[str, torch.Tensor]:
    """Convert ViT state dict to HuggingFace GPT-2 format.

    Handles:
    - Key renaming (blocks.X -> h.X)
    - Weight transposition for Conv1D compatibility
    - LayerNorm key mapping (norm1/norm2 -> ln_1/ln_2)

    Args:
        vit_sd: ViT state dict with keys like "blocks.0.attn.qkv.weight"
        add_prefix: Whether to add "transformer." prefix (default: False).
            Set to False for HuggingFace GPT2Model, True for GPT2LMHeadModel.

    Returns:
        GPT-2 format state dict with keys like "h.0.attn.c_attn.weight"

    Example:
        >>> gpt2_sd = vit_to_gpt2_state_dict(expanded_vit_sd)
        >>> model = GPT2Model(config)
        >>> model.load_state_dict(gpt2_sd, strict=False)  # Missing: wte, wpe
    """
    gpt2_sd = {}

    for vit_key, tensor in vit_sd.items():
        # Handle final norm
        if vit_key == "norm.weight":
            gpt2_key = "transformer.ln_f.weight" if add_prefix else "ln_f.weight"
            gpt2_sd[gpt2_key] = tensor.clone()
            continue
        elif vit_key == "norm.bias":
            gpt2_key = "transformer.ln_f.bias" if add_prefix else "ln_f.bias"
            gpt2_sd[gpt2_key] = tensor.clone()
            continue

        # Handle block keys
        result = _convert_block_key(vit_key, add_prefix)
        if result[0] is None:
            # Unknown key, skip with warning
            print(f"Warning: Skipping unknown ViT key: {vit_key}")
            continue

        gpt2_key, needs_transpose = result

        if needs_transpose and tensor.dim() == 2:
            # Transpose weight for Conv1D format: (out, in) -> (in, out)
            gpt2_sd[gpt2_key] = tensor.T.contiguous()
        else:
            gpt2_sd[gpt2_key] = tensor.clone()

    return gpt2_sd


def gpt2_to_vit_state_dict(
    gpt2_sd: Dict[str, torch.Tensor],
    strip_prefix: bool = True,
) -> Dict[str, torch.Tensor]:
    """Convert HuggingFace GPT-2 state dict to ViT format.

    The reverse of vit_to_gpt2_state_dict.

    Args:
        gpt2_sd: GPT-2 state dict with keys like "transformer.h.0.attn.c_attn.weight"
        strip_prefix: Whether to strip "transformer." prefix (default: True)

    Returns:
        ViT format state dict with keys like "blocks.0.attn.qkv.weight"
    """
    # Build reverse mapping
    gpt2_to_vit_map = {v: k for k, v in VIT_TO_GPT2_COMPONENT_MAP.items()}

    vit_sd = {}

    for gpt2_key, tensor in gpt2_sd.items():
        # Strip prefix if present
        key = gpt2_key
        if strip_prefix and key.startswith("transformer."):
            key = key[len("transformer.") :]

        # Handle final norm
        if key == "ln_f.weight":
            vit_sd["norm.weight"] = tensor.clone()
            continue
        elif key == "ln_f.bias":
            vit_sd["norm.bias"] = tensor.clone()
            continue

        # Handle block keys
        match = re.match(r"h\.(\d+)\.(.+)", key)
        if not match:
            # Skip non-block keys (embeddings, etc.)
            continue

        block_idx = match.group(1)
        gpt2_component = match.group(2)

        if gpt2_component not in gpt2_to_vit_map:
            print(f"Warning: Skipping unknown GPT-2 component: {gpt2_component}")
            continue

        vit_component = gpt2_to_vit_map[gpt2_component]
        vit_key = f"blocks.{block_idx}.{vit_component}"

        needs_transpose = gpt2_component in TRANSPOSE_COMPONENTS

        if needs_transpose and tensor.dim() == 2:
            # Transpose back from Conv1D format: (in, out) -> (out, in)
            vit_sd[vit_key] = tensor.T.contiguous()
        else:
            vit_sd[vit_key] = tensor.clone()

    return vit_sd


def get_gpt2_config_from_vit(
    vit_cfg: Dict,
) -> Dict:
    """Generate HuggingFace GPT2Config parameters from ViT config.

    Args:
        vit_cfg: ViT config with embed_dim, depth, mlp_ratio, num_heads

    Returns:
        Dict with GPT2Config parameters
    """
    embed_dim = vit_cfg["embed_dim"]
    depth = vit_cfg["depth"]
    num_heads = vit_cfg.get("num_heads", embed_dim // 64)
    mlp_ratio = vit_cfg.get("mlp_ratio", 4)

    return {
        "n_embd": embed_dim,
        "n_layer": depth,
        "n_head": num_heads,
        "n_inner": int(embed_dim * mlp_ratio),  # MLP hidden dim
        # Other common defaults
        "activation_function": "gelu_new",
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0,
    }
