#!/usr/bin/env python3
"""
Command-line tool to shuffle weights in ViT checkpoints.
Supports timm-style ViT models with combined QKV weights.
"""

import argparse
import torch
import copy
from pathlib import Path


def load_checkpoint(checkpoint_path):
    """Load checkpoint and return state dict."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get state dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    print(f"Checkpoint has {len(state_dict)} keys")
    return checkpoint, state_dict


def detect_model_config(state_dict):
    """Detect model configuration from state dict."""
    # Try to detect from attention weights first
    sample_key = "blocks.0.attn.qkv.weight"
    if sample_key in state_dict:
        qkv_weight = state_dict[sample_key]
        embed_dim = qkv_weight.shape[1]
        print(f"Detected embed_dim: {embed_dim} (from attention)")

        # Count blocks from attention
        block_indices = set()
        for key in state_dict.keys():
            if key.startswith("blocks.") and ".attn.qkv." in key:
                block_idx = int(key.split(".")[1])
                block_indices.add(block_idx)
    else:
        # Try to detect from MLP weights (for MLP-only checkpoints)
        mlp_key = "blocks.0.mlp.fc2.weight"
        if mlp_key in state_dict:
            mlp_weight = state_dict[mlp_key]
            embed_dim = mlp_weight.shape[0]  # fc2 output dimension is embed_dim
            print(f"Detected embed_dim: {embed_dim} (from MLP)")

            # Count blocks from MLP
            block_indices = set()
            for key in state_dict.keys():
                if key.startswith("blocks.") and ".mlp." in key:
                    block_idx = int(key.split(".")[1])
                    block_indices.add(block_idx)
        else:
            raise ValueError(
                "Could not detect model config: no attention or MLP weights found"
            )

    # Infer number of heads
    if embed_dim == 192:
        num_heads = 3  # ViT-Tiny
    elif embed_dim == 384:
        num_heads = 6  # ViT-Small
    elif embed_dim == 768:
        num_heads = 12  # ViT-Base
    else:
        num_heads = embed_dim // 64

    print(f"Inferred num_heads: {num_heads}")

    num_blocks = len(block_indices)
    print(f"Found {num_blocks} transformer blocks")

    return embed_dim, num_heads, num_blocks


def shuffle_ffn_only(
    state_dict, seed=42, validate=True, assert_preserve_stats=True, atol=1e-5
):
    """Shuffle only FFN (MLP) weights, not attention."""
    import random

    print("Shuffling FFN/MLP weights only...")

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

    def check_and_print(name, pre_tensor, post_tensor):
        pre_mean, pre_std = pre_tensor.mean().item(), pre_tensor.std().item()
        post_mean, post_std = post_tensor.mean().item(), post_tensor.std().item()
        print(
            f"    {name} mean/std: {pre_mean:.4f} / {pre_std:.4f} → {post_mean:.4f} / {post_std:.4f}"
        )
        if assert_preserve_stats:
            assert abs(pre_mean - post_mean) < atol, f"{name} mean changed too much"
            assert abs(pre_std - post_std) < atol, f"{name} std changed too much"

    # Find all block indices
    block_indices = set()
    for key in state_dict.keys():
        if key.startswith("blocks.") and ".mlp." in key:
            block_idx = int(key.split(".")[1])
            block_indices.add(block_idx)

    block_indices = sorted(block_indices)

    for block_idx in block_indices:
        print(
            f"\n  Shuffling FFN weights in block {block_idx + 1}/{len(block_indices)}"
        )

        # First MLP layer (expanding dimension) - fc1
        original_mlp_fc1_w = state_dict[f"blocks.{block_idx}.mlp.fc1.weight"].clone()
        original_mlp_fc1_b = state_dict[f"blocks.{block_idx}.mlp.fc1.bias"].clone()

        mlp_fc1_w_shuf, mlp_fc1_b_shuf = shuffle_pair(
            state_dict[f"blocks.{block_idx}.mlp.fc1.weight"],
            state_dict[f"blocks.{block_idx}.mlp.fc1.bias"],
        )
        state_dict[f"blocks.{block_idx}.mlp.fc1.weight"] = mlp_fc1_w_shuf
        state_dict[f"blocks.{block_idx}.mlp.fc1.bias"] = mlp_fc1_b_shuf

        # Second MLP layer (projecting back to model dimension) - fc2
        original_mlp_fc2_w = state_dict[f"blocks.{block_idx}.mlp.fc2.weight"].clone()
        original_mlp_fc2_b = state_dict[f"blocks.{block_idx}.mlp.fc2.bias"].clone()

        mlp_fc2_w_shuf, mlp_fc2_b_shuf = shuffle_pair(
            state_dict[f"blocks.{block_idx}.mlp.fc2.weight"],
            state_dict[f"blocks.{block_idx}.mlp.fc2.bias"],
        )
        state_dict[f"blocks.{block_idx}.mlp.fc2.weight"] = mlp_fc2_w_shuf
        state_dict[f"blocks.{block_idx}.mlp.fc2.bias"] = mlp_fc2_b_shuf

        if validate:
            print("    Validating statistics preserved:")
            check_and_print("MLP-FC1-W", original_mlp_fc1_w, mlp_fc1_w_shuf)
            check_and_print("MLP-FC1-B", original_mlp_fc1_b, mlp_fc1_b_shuf)
            check_and_print("MLP-FC2-W", original_mlp_fc2_w, mlp_fc2_w_shuf)
            check_and_print("MLP-FC2-B", original_mlp_fc2_b, mlp_fc2_b_shuf)

    print("\nFFN weights successfully shuffled")
    return state_dict


def shuffle_attention_weights_timm(
    state_dict,
    num_heads,
    seed=42,
    validate=True,
    assert_preserve_stats=True,
    atol=1e-5,
):
    """Shuffle attention Q/K/V per head for timm-style ViT checkpoints.

    Supports both combined `qkv` Linear (out=in=embed_dim, out_features=3*embed_dim)
    and separate `q`, `k`, `v` Linear layers. For each transformer block and for
    each of Q, K, V, we shuffle the weights within each head-specific sub-matrix
    independently (and the corresponding bias segment if present).
    """
    import random

    torch.manual_seed(seed)
    random.seed(seed)

    def stats(t):
        return (t.mean().item(), t.std().item())

    def shuffle_inplace_tensor(tensor):
        flat = tensor.flatten()
        perm = torch.randperm(flat.numel())
        tensor.copy_(flat[perm].view_as(tensor))

    def shuffle_qkv_combined(qkv_w, qkv_b, embed_dim):
        head_dim = embed_dim // num_heads

        # Shuffle weights per head, per (Q,K,V)
        for part_idx in range(3):  # 0:Q, 1:K, 2:V
            start = part_idx * embed_dim
            for h in range(num_heads):
                rs = start + h * head_dim
                re = start + (h + 1) * head_dim
                sub_w = qkv_w[rs:re, :]  # shape: [head_dim, embed_dim]
                shuffle_inplace_tensor(sub_w)
                if qkv_b is not None:
                    sub_b = qkv_b[rs:re]  # shape: [head_dim]
                    shuffle_inplace_tensor(sub_b)

    def shuffle_single_proj(w, b, embed_dim):
        head_dim = embed_dim // num_heads
        for h in range(num_heads):
            rs = h * head_dim
            re = (h + 1) * head_dim
            sub_w = w[rs:re, :]
            shuffle_inplace_tensor(sub_w)
            if b is not None:
                sub_b = b[rs:re]
                shuffle_inplace_tensor(sub_b)

    # Detect all block indices that have attention weights
    block_indices = set()
    for key in state_dict.keys():
        if key.startswith("blocks.") and (".attn.qkv." in key or ".attn.q." in key):
            try:
                block_indices.add(int(key.split(".")[1]))
            except Exception:
                pass
    block_indices = sorted(block_indices)

    if not block_indices:
        print("No attention weights found to shuffle. Skipping.")
        return state_dict

    print("Shuffling attention Q/K/V per-head...")

    # Infer embed_dim from first available key
    embed_dim = None
    for b in block_indices:
        if f"blocks.{b}.attn.qkv.weight" in state_dict:
            embed_dim = state_dict[f"blocks.{b}.attn.qkv.weight"].shape[1]
            break
        elif f"blocks.{b}.attn.q.weight" in state_dict:
            embed_dim = state_dict[f"blocks.{b}.attn.q.weight"].shape[1]
            break
    if embed_dim is None:
        raise ValueError("Could not infer embed_dim for attention weights")

    for idx, block_idx in enumerate(block_indices):
        print(
            f"\n  Shuffling attention in block {idx + 1}/{len(block_indices)} (index {block_idx})"
        )

        # Combined qkv case
        qkv_w_key = f"blocks.{block_idx}.attn.qkv.weight"
        qkv_b_key = f"blocks.{block_idx}.attn.qkv.bias"

        if qkv_w_key in state_dict:
            w_pre = state_dict[qkv_w_key].clone()
            b_pre = state_dict[qkv_b_key].clone() if qkv_b_key in state_dict else None

            shuffle_qkv_combined(
                state_dict[qkv_w_key], state_dict.get(qkv_b_key), embed_dim
            )

            if validate:
                w_post = state_dict[qkv_w_key]
                wm0, ws0 = stats(w_pre)
                wm1, ws1 = stats(w_post)
                print(
                    f"    QKV-W mean/std: {wm0:.4f} / {ws0:.4f} → {wm1:.4f} / {ws1:.4f}"
                )
                if b_pre is not None:
                    bm0, bs0 = stats(b_pre)
                    bm1, bs1 = stats(state_dict[qkv_b_key])
                    print(
                        f"    QKV-B mean/std: {bm0:.4f} / {bs0:.4f} → {bm1:.4f} / {bs1:.4f}"
                    )
                if assert_preserve_stats:
                    assert abs(wm0 - wm1) < atol, "QKV weight mean changed too much"
                    assert abs(ws0 - ws1) < atol, "QKV weight std changed too much"
                    if b_pre is not None:
                        assert abs(bm0 - bm1) < atol, "QKV bias mean changed too much"
                        assert abs(bs0 - bs1) < atol, "QKV bias std changed too much"
            continue

        # Separate q, k, v case
        for part in ["q", "k", "v"]:
            w_key = f"blocks.{block_idx}.attn.{part}.weight"
            b_key = f"blocks.{block_idx}.attn.{part}.bias"
            if w_key not in state_dict:
                continue

            w_pre = state_dict[w_key].clone()
            b_pre = state_dict[b_key].clone() if b_key in state_dict else None

            shuffle_single_proj(state_dict[w_key], state_dict.get(b_key), embed_dim)

            if validate:
                w_post = state_dict[w_key]
                wm0, ws0 = stats(w_pre)
                wm1, ws1 = stats(w_post)
                print(
                    f"    {part.upper()}-W mean/std: {wm0:.4f} / {ws0:.4f} → {wm1:.4f} / {ws1:.4f}"
                )
                if b_pre is not None:
                    bm0, bs0 = stats(b_pre)
                    bm1, bs1 = stats(state_dict[b_key])
                    print(
                        f"    {part.upper()}-B mean/std: {bm0:.4f} / {bs0:.4f} → {bm1:.4f} / {bs1:.4f}"
                    )
                if assert_preserve_stats:
                    assert abs(wm0 - wm1) < atol, (
                        f"{part.upper()} weight mean changed too much"
                    )
                    assert abs(ws0 - ws1) < atol, (
                        f"{part.upper()} weight std changed too much"
                    )
                    if b_pre is not None:
                        assert abs(bm0 - bm1) < atol, (
                            f"{part.upper()} bias mean changed too much"
                        )
                        assert abs(bs0 - bs1) < atol, (
                            f"{part.upper()} bias std changed too much"
                        )

    print("\nAttention weights successfully shuffled")
    return state_dict


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle weights in ViT checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Shuffle only attention weights
  python shuffle_checkpoint.py path/to/checkpoint.pth --shuffle attention
  
  # Shuffle only FFN weights
  python shuffle_checkpoint.py path/to/checkpoint.pth --shuffle ffn
  
  # Shuffle both attention and FFN weights
  python shuffle_checkpoint.py path/to/checkpoint.pth --shuffle both
  
  # Specify output path and seed
  python shuffle_checkpoint.py path/to/checkpoint.pth --shuffle both --output shuffled.pth --seed 123
        """,
    )

    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the checkpoint file to shuffle",
    )

    parser.add_argument(
        "--shuffle",
        type=str,
        choices=["attention", "ffn", "both"],
        required=True,
        help="Which weights to shuffle: 'attention', 'ffn', or 'both'",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for shuffled checkpoint (default: adds '_shuffled_<type>' suffix)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation of mean/std preservation",
    )

    parser.add_argument(
        "--no-assert",
        action="store_true",
        help="Don't assert that statistics are preserved (just warn)",
    )

    args = parser.parse_args()

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint, state_dict = load_checkpoint(checkpoint_path)

    # Detect model configuration
    embed_dim, num_heads, num_blocks = detect_model_config(state_dict)

    # Make a copy of state dict for shuffling
    state_dict_shuffled = copy.deepcopy(state_dict)

    # Perform shuffling based on choice
    print(f"\n{'=' * 80}")
    print(f"Shuffling: {args.shuffle.upper()}")
    print(f"Seed: {args.seed}")
    print(f"{'=' * 80}\n")

    validate = not args.no_validate
    assert_preserve_stats = not args.no_assert

    if args.shuffle == "attention":
        state_dict_shuffled = shuffle_attention_weights_timm(
            state_dict_shuffled,
            num_heads=num_heads,
            seed=args.seed,
            validate=validate,
            assert_preserve_stats=assert_preserve_stats,
        )
    elif args.shuffle == "ffn":
        state_dict_shuffled = shuffle_ffn_only(
            state_dict_shuffled,
            seed=args.seed,
            validate=validate,
            assert_preserve_stats=assert_preserve_stats,
        )
    elif args.shuffle == "both":
        # Shuffle attention first, then FFN (using same seed)
        state_dict_shuffled = shuffle_attention_weights_timm(
            state_dict_shuffled,
            num_heads=num_heads,
            seed=args.seed,
            validate=validate,
            assert_preserve_stats=assert_preserve_stats,
        )
        state_dict_shuffled = shuffle_ffn_only(
            state_dict_shuffled,
            seed=args.seed,
            validate=validate,
            assert_preserve_stats=assert_preserve_stats,
        )

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Add suffix to original filename
        suffix = f"_shuffled_{args.shuffle}"
        output_path = (
            checkpoint_path.parent
            / f"{checkpoint_path.stem}{suffix}{checkpoint_path.suffix}"
        )

    # Save shuffled checkpoint
    print(f"\n{'=' * 80}")
    print(f"Saving shuffled checkpoint to: {output_path}")
    print(f"{'=' * 80}\n")

    # Update the state dict in the original checkpoint structure
    if "model" in checkpoint:
        checkpoint["model"] = state_dict_shuffled
    elif "state_dict" in checkpoint:
        checkpoint["state_dict"] = state_dict_shuffled
    else:
        checkpoint = state_dict_shuffled

    torch.save(checkpoint, output_path)
    print("✅ Successfully saved shuffled checkpoint!")
    print(f"   Original: {checkpoint_path}")
    print(f"   Shuffled: {output_path}")
    print(f"   Shuffle type: {args.shuffle}")
    print(f"   Seed: {args.seed}")


if __name__ == "__main__":
    main()
