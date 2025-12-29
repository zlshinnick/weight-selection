"""Utility functions for weight expansion.

Provides checkpoint loading, key parsing, and parameter classification.
"""

import re
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch


class ParamType(Enum):
    """Enumeration of parameter types in ViT transformer blocks."""
    QKV_WEIGHT = auto()
    QKV_BIAS = auto()
    PROJ_WEIGHT = auto()
    PROJ_BIAS = auto()
    FC1_WEIGHT = auto()
    FC1_BIAS = auto()
    FC2_WEIGHT = auto()
    FC2_BIAS = auto()
    NORM1_WEIGHT = auto()
    NORM1_BIAS = auto()
    NORM2_WEIGHT = auto()
    NORM2_BIAS = auto()
    FINAL_NORM_WEIGHT = auto()
    FINAL_NORM_BIAS = auto()
    UNKNOWN = auto()


# Regex pattern for matching block parameter keys
BLOCK_KEY_PATTERN = re.compile(r"blocks\.(\d+)\.(.+)")

# Mapping from key suffix to parameter type
PARAM_TYPE_MAP = {
    "attn.qkv.weight": ParamType.QKV_WEIGHT,
    "attn.qkv.bias": ParamType.QKV_BIAS,
    "attn.proj.weight": ParamType.PROJ_WEIGHT,
    "attn.proj.bias": ParamType.PROJ_BIAS,
    "mlp.fc1.weight": ParamType.FC1_WEIGHT,
    "mlp.fc1.bias": ParamType.FC1_BIAS,
    "mlp.fc2.weight": ParamType.FC2_WEIGHT,
    "mlp.fc2.bias": ParamType.FC2_BIAS,
    "norm1.weight": ParamType.NORM1_WEIGHT,
    "norm1.bias": ParamType.NORM1_BIAS,
    "norm2.weight": ParamType.NORM2_WEIGHT,
    "norm2.bias": ParamType.NORM2_BIAS,
}


def load_checkpoint(path: str | Path) -> Tuple[Dict[str, torch.Tensor], Optional[Dict], str]:
    """Load a checkpoint and extract the state dict.
    
    Handles common checkpoint wrapper formats (model, state_dict, etc.).
    
    Args:
        path: Path to the checkpoint file.
        
    Returns:
        Tuple of (state_dict, full_checkpoint, wrapper_key):
            - state_dict: The model state dictionary
            - full_checkpoint: The full checkpoint dict (or None if raw state_dict)
            - wrapper_key: The key used to wrap the state_dict (empty if raw)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    obj = torch.load(path, map_location="cpu", weights_only=False)
    
    if isinstance(obj, dict):
        # Check common wrapper keys
        for key in ("state_dict", "model_state_dict", "model", "module_state_dict", "weights"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key], obj, key
        
        # Check if it's a raw state_dict (all values are tensors)
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj, None, ""
        
        # Check for nested dict with tensors
        for k, v in obj.items():
            if isinstance(v, dict) and v and all(isinstance(vv, torch.Tensor) for vv in v.values()):
                return v, obj, k
    
    # Fallback: assume raw state_dict
    if isinstance(obj, dict):
        return obj, None, ""
    
    raise ValueError("Unsupported checkpoint format")


def parse_block_key(key: str) -> Optional[Tuple[int, str]]:
    """Parse a parameter key to extract block index and component path.
    
    Args:
        key: Parameter key like "blocks.0.attn.qkv.weight"
        
    Returns:
        Tuple of (block_index, component_path) or None if not a block parameter.
        Example: (0, "attn.qkv.weight")
    """
    match = BLOCK_KEY_PATTERN.match(key)
    if match:
        return int(match.group(1)), match.group(2)
    return None


def get_param_type(key: str) -> ParamType:
    """Determine the parameter type from a key.
    
    Args:
        key: Parameter key like "blocks.0.attn.qkv.weight" or "norm.weight"
        
    Returns:
        ParamType enum value.
    """
    # Handle final norm (not in a block)
    if key == "norm.weight":
        return ParamType.FINAL_NORM_WEIGHT
    if key == "norm.bias":
        return ParamType.FINAL_NORM_BIAS
    
    # Parse block key
    parsed = parse_block_key(key)
    if parsed is None:
        return ParamType.UNKNOWN
    
    _, component = parsed
    return PARAM_TYPE_MAP.get(component, ParamType.UNKNOWN)


def is_block_key(key: str) -> bool:
    """Check if a key belongs to a transformer block."""
    return parse_block_key(key) is not None


def is_weight_key(key: str) -> bool:
    """Check if a key is for a weight matrix (not bias or norm)."""
    ptype = get_param_type(key)
    return ptype in (
        ParamType.QKV_WEIGHT,
        ParamType.PROJ_WEIGHT,
        ParamType.FC1_WEIGHT,
        ParamType.FC2_WEIGHT,
    )


def is_bias_key(key: str) -> bool:
    """Check if a key is for a bias vector (not LayerNorm bias)."""
    ptype = get_param_type(key)
    return ptype in (
        ParamType.QKV_BIAS,
        ParamType.PROJ_BIAS,
        ParamType.FC1_BIAS,
        ParamType.FC2_BIAS,
    )


def is_layernorm_key(key: str) -> bool:
    """Check if a key is for a LayerNorm parameter."""
    ptype = get_param_type(key)
    return ptype in (
        ParamType.NORM1_WEIGHT,
        ParamType.NORM1_BIAS,
        ParamType.NORM2_WEIGHT,
        ParamType.NORM2_BIAS,
        ParamType.FINAL_NORM_WEIGHT,
        ParamType.FINAL_NORM_BIAS,
    )


def is_layernorm_weight(key: str) -> bool:
    """Check if a key is for LayerNorm gamma (weight)."""
    ptype = get_param_type(key)
    return ptype in (
        ParamType.NORM1_WEIGHT,
        ParamType.NORM2_WEIGHT,
        ParamType.FINAL_NORM_WEIGHT,
    )


def is_layernorm_bias(key: str) -> bool:
    """Check if a key is for LayerNorm beta (bias)."""
    ptype = get_param_type(key)
    return ptype in (
        ParamType.NORM1_BIAS,
        ParamType.NORM2_BIAS,
        ParamType.FINAL_NORM_BIAS,
    )


def filter_block_params(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Filter state dict to only include transformer block parameters and final norm.
    
    Args:
        state_dict: Full state dictionary.
        
    Returns:
        State dict containing only block parameters (blocks.*) and final norm.
    """
    result = {}
    for key, value in state_dict.items():
        if is_block_key(key) or key in ("norm.weight", "norm.bias"):
            result[key] = value
    return result


def get_block_indices(state_dict: Dict[str, torch.Tensor]) -> list[int]:
    """Extract sorted list of block indices present in state dict.
    
    Args:
        state_dict: State dictionary.
        
    Returns:
        Sorted list of block indices.
    """
    indices = set()
    for key in state_dict.keys():
        parsed = parse_block_key(key)
        if parsed is not None:
            indices.add(parsed[0])
    return sorted(indices)


def make_block_key(block_idx: int, component: str) -> str:
    """Construct a block parameter key.
    
    Args:
        block_idx: Block index.
        component: Component path like "attn.qkv.weight".
        
    Returns:
        Full key like "blocks.0.attn.qkv.weight".
    """
    return f"blocks.{block_idx}.{component}"


def validate_config(cfg: Dict[str, Any], required_keys: list[str]) -> None:
    """Validate that a config dict has required keys.
    
    Args:
        cfg: Configuration dictionary.
        required_keys: List of required key names.
        
    Raises:
        ValueError: If any required key is missing.
    """
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

