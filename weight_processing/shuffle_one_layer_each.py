import argparse
import copy
import os
from typing import Dict, List, Optional, Set, Tuple

import torch

# Support running both as part of the package and as a standalone script
try:
    from .subset_weights import (
        collect_layer_indices,
        load_checkpoint,
    )
    from .shuffle_partial_layers import (
        detect_model_config,
        shuffle_attention_weights_timm,
        shuffle_ffn_only,
    )
except Exception:
    import sys as _sys
    import os as _os

    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(__file__)))
    from weight_processing.subset_weights import (
        collect_layer_indices,
        load_checkpoint,
    )
    from weight_processing.shuffle_partial_layers import (
        detect_model_config,
        shuffle_attention_weights_timm,
        shuffle_ffn_only,
    )


StateDict = Dict[str, torch.Tensor]


def derive_output_path(
    input_path: str, output_dir: Optional[str], layer_idx: int, which: str
) -> str:
    base_dir = output_dir or os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    return os.path.join(base_dir, f"{name}_shuf_{which}_layer{layer_idx}{ext}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Given a checkpoint, emit L outputs (L=layers). In each output, only one layer is shuffled."
        ),
    )
    parser.add_argument("input", type=str, help="Path to input .pth checkpoint")
    parser.add_argument(
        "--shuffle",
        type=str,
        choices=["attention", "ffn", "both"],
        default="both",
        help="Which weights to shuffle in the selected single layer",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write outputs (defaults to input file directory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )

    args = parser.parse_args()

    state_dict, wrapper, wrapper_key = load_checkpoint(args.input)
    embed_dim, num_heads, num_blocks = detect_model_config(state_dict)
    all_indices = collect_layer_indices(state_dict)

    print(f"Detected layer indices: {all_indices}")

    for idx in all_indices:
        allowed: Set[int] = {idx}
        sd_copy = copy.deepcopy(state_dict)

        if args.shuffle in ("attention", "both"):
            sd_copy = shuffle_attention_weights_timm(
                sd_copy,
                allowed_blocks=allowed,
                num_heads=num_heads,
                seed=args.seed,
                validate=False,
                assert_preserve_stats=False,
            )
        if args.shuffle in ("ffn", "both"):
            sd_copy = shuffle_ffn_only(
                sd_copy,
                allowed_blocks=allowed,
                seed=args.seed,
                validate=False,
                assert_preserve_stats=False,
            )

        out_path = derive_output_path(args.input, args.output_dir, idx, args.shuffle)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if wrapper is None:
            torch.save(sd_copy, out_path)
        else:
            wrapper_copy = copy.deepcopy(wrapper)
            if wrapper_key:
                wrapper_copy[wrapper_key] = sd_copy
            else:
                wrapper_copy = sd_copy  # type: ignore[assignment]
            torch.save(wrapper_copy, out_path)
        print(f"Wrote per-layer shuffled checkpoint for layer {idx}: {out_path}")


if __name__ == "__main__":
    main()
