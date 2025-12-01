import argparse
import copy
import os
import re
from typing import Dict, List, Optional, Set, Tuple, Union

import torch


StateDict = Dict[str, torch.Tensor]
Checkpoint = Union[StateDict, Dict[str, object]]


LAYER_KEY_REGEXES: List[re.Pattern[str]] = [
    # GPT/Transformer common patterns
    re.compile(
        r"(?:^|\.)transformer\.h\.(\d+)\."
    ),  # e.g., transformer.h.0.attn.c_attn.weight
    re.compile(
        r"(?:^|\.)model\.(?:layers|h)\.(\d+)\."
    ),  # e.g., model.layers.0.self_attn.q_proj.weight
    re.compile(r"(?:^|\.)layers\.(\d+)\."),  # e.g., layers.0.attention.wq.weight
    re.compile(r"(?:^|\.)blocks\.(\d+)\."),  # e.g., blocks.0.attn.qkv.weight
    re.compile(r"(?:^|\.)block\.(\d+)\."),
    re.compile(r"(?:^|\.)h\.(\d+)\."),
    # Encoder/decoder stacks
    re.compile(r"(?:^|\.)encoder\.layers\.(\d+)\."),
    re.compile(r"(?:^|\.)decoder\.layers\.(\d+)\."),
]


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[StateDict, Optional[Dict[str, object]], str]:
    """
    Load a .pth checkpoint and return a tuple of (state_dict, wrapper, wrapper_key).

    - If the file is a plain state_dict: returns (state_dict, None, "").
    - If wrapped (e.g., {"state_dict": ...}): returns (state_dict, wrapper_dict, wrapper_key_used).
    """
    obj = torch.load(checkpoint_path, map_location="cpu")

    # If it's a dict, check common wrapper keys
    if isinstance(obj, dict):
        # Common cases
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "module_state_dict",
            "weights",
        ):
            if key in obj and isinstance(obj[key], dict):
                return obj[key], obj, key

        # If values look like tensors -> likely a raw state_dict
        if obj and all(isinstance(v, (torch.Tensor,)) for v in obj.values()):
            return obj, None, ""

        # Some checkpoints save under nested key paths; try to heuristically locate one dict of tensors
        for k, v in obj.items():
            if (
                isinstance(v, dict)
                and v
                and all(isinstance(vv, torch.Tensor) for vv in v.values())
            ):
                return v, obj, k

    # Fallback: assume it is a raw state_dict
    if isinstance(obj, dict):
        return obj, None, ""

    raise ValueError(
        "Unsupported checkpoint format: expected dict or state_dict-like content"
    )


def detect_layer_index(param_key: str) -> Optional[int]:
    for regex in LAYER_KEY_REGEXES:
        m = regex.search(param_key)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def collect_layer_indices(state_dict: StateDict) -> List[int]:
    indices: Set[int] = set()
    for key in state_dict.keys():
        idx = detect_layer_index(key)
        if idx is not None:
            indices.add(idx)
    return sorted(indices)


def slice_indices_allocation(
    all_indices: List[int], k: int
) -> Tuple[List[int], List[int], List[int]]:
    if k <= 0:
        raise ValueError("k must be positive")
    n = len(all_indices)
    if n == 0:
        raise ValueError("No layer indices detected in checkpoint")
    if k > n:
        raise ValueError(f"k ({k}) exceeds number of detected layers ({n})")

    # Early k: first k
    early = all_indices[:k]

    # Middle k: centered contiguous block
    start = (n - k) // 2
    middle = all_indices[start : start + k]

    # Final k: last k
    final = all_indices[-k:]

    return early, middle, final


def filter_state_dict_by_layers(
    state_dict: StateDict,
    allowed_indices: Set[int],
    keep_nonlayer: bool,
) -> StateDict:
    subset: StateDict = {}
    for key, tensor in state_dict.items():
        idx = detect_layer_index(key)
        if idx is None:
            if keep_nonlayer:
                subset[key] = tensor
            continue
        if idx in allowed_indices:
            subset[key] = tensor
    return subset


def save_subset(
    subset: StateDict,
    original_wrapper: Optional[Dict[str, object]],
    wrapper_key: str,
    output_path: str,
) -> None:
    if original_wrapper is None:
        torch.save(subset, output_path)
        return

    wrapper_copy = copy.deepcopy(original_wrapper)
    if wrapper_key:
        wrapper_copy[wrapper_key] = subset
    else:
        # Unknown wrapper location, fall back to raw save
        wrapper_copy = subset  # type: ignore[assignment]
    torch.save(wrapper_copy, output_path)


def derive_output_paths(
    input_path: str, output_dir: Optional[str], k: int
) -> Tuple[str, str, str]:
    base_dir = output_dir or os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    early = os.path.join(base_dir, f"{name}_early_k{k}{ext}")
    middle = os.path.join(base_dir, f"{name}_middle_k{k}{ext}")
    final = os.path.join(base_dir, f"{name}_final_k{k}{ext}")
    return early, middle, final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a .pth checkpoint into early/middle/final k transformer layers",
    )
    parser.add_argument("input", type=str, help="Path to input .pth checkpoint")
    parser.add_argument("k", type=int, help="Number of layers per split (k)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write outputs (defaults to input file directory)",
    )
    parser.add_argument(
        "--keep-nonlayer",
        action="store_true",
        help="Preserve non-layer parameters (e.g., final norms, heads) in each output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; only print detected layers and would-be outputs",
    )

    args = parser.parse_args()

    state_dict, wrapper, wrapper_key = load_checkpoint(args.input)
    all_indices = collect_layer_indices(state_dict)
    early_idx, middle_idx, final_idx = slice_indices_allocation(all_indices, args.k)

    early_out, middle_out, final_out = derive_output_paths(
        args.input, args.output_dir, args.k
    )

    print(f"Detected {len(all_indices)} layers: {all_indices}")
    print(f"Early k indices:  {early_idx}")
    print(f"Middle k indices: {middle_idx}")
    print(f"Final k indices:  {final_idx}")
    print(f"Will write to:\n  {early_out}\n  {middle_out}\n  {final_out}")

    if args.dry_run:
        return

    early_sd = filter_state_dict_by_layers(
        state_dict, set(early_idx), keep_nonlayer=args.keep_nonlayer
    )
    middle_sd = filter_state_dict_by_layers(
        state_dict, set(middle_idx), keep_nonlayer=args.keep_nonlayer
    )
    final_sd = filter_state_dict_by_layers(
        state_dict, set(final_idx), keep_nonlayer=args.keep_nonlayer
    )

    os.makedirs(os.path.dirname(early_out), exist_ok=True)
    os.makedirs(os.path.dirname(middle_out), exist_ok=True)
    os.makedirs(os.path.dirname(final_out), exist_ok=True)

    save_subset(early_sd, wrapper, wrapper_key, early_out)
    save_subset(middle_sd, wrapper, wrapper_key, middle_out)
    save_subset(final_sd, wrapper, wrapper_key, final_out)

    print("Done.")


if __name__ == "__main__":
    main()
