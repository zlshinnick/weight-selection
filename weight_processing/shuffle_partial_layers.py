import argparse
import copy
import os
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch

from .subset_weights import (
    collect_layer_indices,
    detect_layer_index,
    load_checkpoint,
    slice_indices_allocation,
)


StateDict = Dict[str, torch.Tensor]


def detect_model_config(state_dict: StateDict) -> Tuple[int, int, int]:
    sample_key = "blocks.0.attn.qkv.weight"
    if sample_key in state_dict:
        qkv_weight = state_dict[sample_key]
        embed_dim = qkv_weight.shape[1]

        block_indices: Set[int] = set()
        for key in state_dict.keys():
            if key.startswith("blocks.") and ".attn.qkv." in key:
                block_idx = int(key.split(".")[1])
                block_indices.add(block_idx)
    else:
        mlp_key = "blocks.0.mlp.fc2.weight"
        if mlp_key in state_dict:
            mlp_weight = state_dict[mlp_key]
            embed_dim = mlp_weight.shape[0]

            block_indices = set()
            for key in state_dict.keys():
                if key.startswith("blocks.") and ".mlp." in key:
                    block_idx = int(key.split(".")[1])
                    block_indices.add(block_idx)
        else:
            raise ValueError(
                "Could not detect model config: no attention or MLP weights found"
            )

    if embed_dim == 192:
        num_heads = 3
    elif embed_dim == 384:
        num_heads = 6
    elif embed_dim == 768:
        num_heads = 12
    else:
        num_heads = embed_dim // 64

    num_blocks = len(block_indices)
    return embed_dim, num_heads, num_blocks


def shuffle_ffn_only(
    state_dict: StateDict,
    allowed_blocks: Set[int],
    seed: int = 42,
    validate: bool = True,
    assert_preserve_stats: bool = True,
    atol: float = 1e-5,
) -> StateDict:
    import random

    torch.manual_seed(seed)
    random.seed(seed)

    def shuffle_pair(w, b=None):
        flat_w = w.flatten()
        perm = torch.randperm(flat_w.numel())
        w_shuffled = flat_w[perm].view_as(w)
        if b is not None:
            b_shuffled = b[torch.randperm(b.numel())]
            return w_shuffled, b_shuffled
        else:
            return w_shuffled

    def stats(t):
        return (t.mean().item(), t.std().item())

    for block_idx in sorted(allowed_blocks):
        fc1_w_key = f"blocks.{block_idx}.mlp.fc1.weight"
        fc1_b_key = f"blocks.{block_idx}.mlp.fc1.bias"
        fc2_w_key = f"blocks.{block_idx}.mlp.fc2.weight"
        fc2_b_key = f"blocks.{block_idx}.mlp.fc2.bias"
        if fc1_w_key not in state_dict or fc2_w_key not in state_dict:
            continue

        w1_pre = state_dict[fc1_w_key].clone()
        b1_pre = state_dict[fc1_b_key].clone() if fc1_b_key in state_dict else None
        w2_pre = state_dict[fc2_w_key].clone()
        b2_pre = state_dict[fc2_b_key].clone() if fc2_b_key in state_dict else None

        w1_post, b1_post = shuffle_pair(
            state_dict[fc1_w_key], state_dict.get(fc1_b_key)
        )
        state_dict[fc1_w_key] = w1_post
        if fc1_b_key in state_dict:
            state_dict[fc1_b_key] = b1_post

        w2_post, b2_post = shuffle_pair(
            state_dict[fc2_w_key], state_dict.get(fc2_b_key)
        )
        state_dict[fc2_w_key] = w2_post
        if fc2_b_key in state_dict:
            state_dict[fc2_b_key] = b2_post

        if validate:
            wm0, ws0 = stats(w1_pre)
            wm1, ws1 = stats(state_dict[fc1_w_key])
            bm0, bs0 = stats(b1_pre) if b1_pre is not None else (None, None)
            bm1, bs1 = (
                stats(state_dict[fc1_b_key]) if b1_pre is not None else (None, None)
            )
            # Simple print validation; optional asserts
            if assert_preserve_stats:
                assert abs(wm0 - wm1) < atol and abs(ws0 - ws1) < atol
                if b1_pre is not None:
                    assert abs(bm0 - bm1) < atol and abs(bs0 - bs1) < atol

            wm0, ws0 = stats(w2_pre)
            wm1, ws1 = stats(state_dict[fc2_w_key])
            bm0, bs0 = stats(b2_pre) if b2_pre is not None else (None, None)
            bm1, bs1 = (
                stats(state_dict[fc2_b_key]) if b2_pre is not None else (None, None)
            )
            if assert_preserve_stats:
                assert abs(wm0 - wm1) < atol and abs(ws0 - ws1) < atol
                if b2_pre is not None:
                    assert abs(bm0 - bm1) < atol and abs(bs0 - bs1) < atol

    return state_dict


def shuffle_attention_weights_timm(
    state_dict: StateDict,
    allowed_blocks: Set[int],
    num_heads: int,
    seed: int = 42,
    validate: bool = True,
    assert_preserve_stats: bool = True,
    atol: float = 1e-5,
) -> StateDict:
    import random

    torch.manual_seed(seed)
    random.seed(seed)

    def shuffle_inplace_tensor(tensor):
        flat = tensor.flatten()
        perm = torch.randperm(flat.numel())
        tensor.copy_(flat[perm].view_as(tensor))

    # Infer embed_dim
    embed_dim = None
    for b in sorted(allowed_blocks):
        k = f"blocks.{b}.attn.qkv.weight"
        if k in state_dict:
            embed_dim = state_dict[k].shape[1]
            break
        k = f"blocks.{b}.attn.q.weight"
        if k in state_dict:
            embed_dim = state_dict[k].shape[1]
            break
    if embed_dim is None:
        # Fallback: try any block
        for key in state_dict.keys():
            if key.endswith("attn.qkv.weight") or key.endswith("attn.q.weight"):
                embed_dim = state_dict[key].shape[1]
                break
    if embed_dim is None:
        raise ValueError("Could not infer embed_dim for attention weights")

    head_dim = embed_dim // num_heads

    for block_idx in sorted(allowed_blocks):
        qkv_w_key = f"blocks.{block_idx}.attn.qkv.weight"
        qkv_b_key = f"blocks.{block_idx}.attn.qkv.bias"
        if qkv_w_key in state_dict:
            w = state_dict[qkv_w_key]
            b = state_dict.get(qkv_b_key)
            for part in range(3):
                start = part * embed_dim
                for h in range(num_heads):
                    rs = start + h * head_dim
                    re = start + (h + 1) * head_dim
                    sub_w = w[rs:re, :]
                    shuffle_inplace_tensor(sub_w)
                    if b is not None:
                        sub_b = b[rs:re]
                        shuffle_inplace_tensor(sub_b)
            continue

        # Separate q, k, v
        for part in ["q", "k", "v"]:
            w_key = f"blocks.{block_idx}.attn.{part}.weight"
            b_key = f"blocks.{block_idx}.attn.{part}.bias"
            if w_key not in state_dict:
                continue
            w = state_dict[w_key]
            b = state_dict.get(b_key)
            for h in range(num_heads):
                rs = h * head_dim
                re = (h + 1) * head_dim
                sub_w = w[rs:re, :]
                shuffle_inplace_tensor(sub_w)
                if b is not None:
                    sub_b = b[rs:re]
                    shuffle_inplace_tensor(sub_b)

    return state_dict


def derive_output_paths(
    input_path: str, output_dir: Optional[str], k: int
) -> Tuple[str, str, str]:
    base_dir = output_dir or os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    early = os.path.join(base_dir, f"{name}_shuf_early_k{k}{ext}")
    middle = os.path.join(base_dir, f"{name}_shuf_middle_k{k}{ext}")
    final = os.path.join(base_dir, f"{name}_shuf_final_k{k}{ext}")
    return early, middle, final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy checkpoint and shuffle only first/middle/final k layers",
    )
    parser.add_argument("input", type=str, help="Path to input .pth checkpoint")
    parser.add_argument("k", type=int, help="Number of layers per split (k)")
    parser.add_argument(
        "--shuffle",
        type=str,
        choices=["attention", "ffn", "both"],
        default="both",
        help="Which weights to shuffle in selected layers",
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
    early_idx, middle_idx, final_idx = slice_indices_allocation(all_indices, args.k)

    print(f"Detected {len(all_indices)} layers: {all_indices}")
    print(
        f"Shuffle target indices -> early: {early_idx}, middle: {middle_idx}, final: {final_idx}"
    )

    early_out, middle_out, final_out = derive_output_paths(
        args.input, args.output_dir, args.k
    )

    for tag, idxs, out_path in (
        ("early", set(early_idx), early_out),
        ("middle", set(middle_idx), middle_out),
        ("final", set(final_idx), final_out),
    ):
        sd_copy = copy.deepcopy(state_dict)
        if args.shuffle in ("attention", "both"):
            sd_copy = shuffle_attention_weights_timm(
                sd_copy,
                allowed_blocks=idxs,
                num_heads=num_heads,
                seed=args.seed,
                validate=False,
                assert_preserve_stats=False,
            )
        if args.shuffle in ("ffn", "both"):
            sd_copy = shuffle_ffn_only(
                sd_copy,
                allowed_blocks=idxs,
                seed=args.seed,
                validate=False,
                assert_preserve_stats=False,
            )

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
        print(f"Wrote {tag} shuffled checkpoint: {out_path}")


if __name__ == "__main__":
    main()
