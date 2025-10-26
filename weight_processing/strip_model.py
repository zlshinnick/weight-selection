#!/usr/bin/env python3
"""
Convert and clean pretrain checkpoints from .pt to .pth format.
Removes MLM-specific weights and untrained components.
"""

import sys
from pathlib import Path

# Add parent directory to path so symbolic_pretrain can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def convert_and_clean_checkpoint(input_path, output_path):
    """
    Convert .pt checkpoint to clean .pth format.

    Steps:
    1. Load the .pt checkpoint
    2. Extract model_state (from symbolic_pretrain.py format)
    3. Strip wrapper prefixes (vit., tok., pos.)
    4. Remove untrained components (patch_embed, pos_embed, head)
    5. Save as clean .pth file
    """
    print(f"\nProcessing: {input_path.name}")
    print("-" * 80)

    # Load checkpoint
    print("  Loading checkpoint...")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    print(f"  Checkpoint keys: {list(checkpoint.keys())}")

    # Extract model state
    if "model_state" in checkpoint:
        model_state_dict = checkpoint["model_state"]
    elif "model" in checkpoint:
        model_state_dict = checkpoint["model"]
    else:
        raise ValueError(f"Unknown checkpoint format: {checkpoint.keys()}")

    print(f"  Original model keys: {len(model_state_dict)}")

    # Clean the state dict
    cleaned_state_dict = {}
    removed_keys = []

    # Keys to remove (untrained or MLM-specific)
    keys_to_skip = {
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "pos_embed",
        "head.weight",
        "head.bias",
        "tok.weight",  # MLM token embeddings
        "pos.weight",  # MLM position embeddings
    }

    for key, value in model_state_dict.items():
        # Skip MLM-specific wrapper keys
        if key in ["tok.weight", "pos.weight"]:
            removed_keys.append(f"{key} (MLM-specific)")
            continue

        # Strip 'vit.' prefix from transformer weights
        if key.startswith("vit."):
            new_key = key[4:]  # Remove 'vit.' prefix

            # Check if this is a key we want to skip
            if new_key in keys_to_skip:
                removed_keys.append(f"{key} -> {new_key} (untrained)")
                continue

            cleaned_state_dict[new_key] = value

        # Skip untrained components
        elif key in keys_to_skip:
            removed_keys.append(f"{key} (untrained)")
            continue

        else:
            cleaned_state_dict[key] = value

    print(f"\n  Cleaned state dict: {len(cleaned_state_dict)} keys")
    print(f"  Removed {len(removed_keys)} keys:")
    for removed in removed_keys[:5]:  # Show first 5
        print(f"    - {removed}")
    if len(removed_keys) > 5:
        print(f"    ... and {len(removed_keys) - 5} more")

    # Save as clean .pth file
    clean_checkpoint = {"model": cleaned_state_dict}
    print(f"\n  Saving to {output_path.name}...")
    torch.save(clean_checkpoint, output_path)

    # Check file sizes
    input_size = input_path.stat().st_size / (1024 * 1024)  # MB
    output_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"  ✓ Saved: {input_size:.1f} MB -> {output_size:.1f} MB")

    return len(cleaned_state_dict), len(removed_keys)


def main():
    # Directory containing new pretrain checkpoints
    checkpoint_dir = Path(
        "spt_models/dyck-mlm/v7_shuffled/close-only"
    )

    # Find all .pt files
    pt_files = sorted(checkpoint_dir.glob("ckpt_step_*.pt"))

    if not pt_files:
        print(f"No checkpoint .pt files found in {checkpoint_dir}")
        return

    print("=" * 80)
    print("CONVERTING AND CLEANING PRETRAIN CHECKPOINTS")
    print("=" * 80)
    print(f"\nFound {len(pt_files)} checkpoint(s) to process")
    print("\nOperations:")
    print("  1. Extract model weights from .pt checkpoint")
    print("  2. Strip 'vit.' prefix from transformer weights")
    print("  3. Remove MLM-specific weights (tok, pos embeddings)")
    print("  4. Remove untrained components (patch_embed, pos_embed, head)")
    print("  5. Save as clean .pth file for fine-tuning")

    results = {}
    failures = []
    for pt_file in pt_files:
        # Create output filename (replace .pt with .pth, add _no_embed suffix)
        base_name = pt_file.stem  # e.g., "ckpt_step_030000"
        pth_file = checkpoint_dir / f"{base_name}_no_embed.pth"

        try:
            num_keys, num_removed = convert_and_clean_checkpoint(pt_file, pth_file)
            results[pth_file.name] = (num_keys, num_removed)
        except Exception as e:
            print(f"\n  ✗ Error processing {pt_file.name}: {e}\n")
            failures.append((pt_file.name, str(e)))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        print("\n✓ Successfully processed:")
        for name, (num_keys, num_removed) in results.items():
            print(f"  ✓ {name}")
            print(f"      {num_keys} keys kept, {num_removed} keys removed")

    if failures:
        print("\n✗ Failed to process:")
        for name, error in failures:
            print(f"  ✗ {name}")
            print(f"      Error: {error}")

    if results and not failures:
        print("\n✓ All checkpoints processed successfully!")
        print(f"\nCleaned checkpoints saved in: {checkpoint_dir}")
        print("\nThese .pth files contain ONLY the trained transformer weights.")
        print("patch_embed, pos_embed, head, and MLM components removed.")
        print("\nReady for fine-tuning! Use the *_no_embed.pth files.")
    elif results:
        print(f"\n⚠ {len(results)}/{len(pt_files)} checkpoints processed successfully.")
        print(f"Cleaned checkpoints saved in: {checkpoint_dir}")
    else:
        print("\n✗ All checkpoints failed to process. Please check the errors above.")


if __name__ == "__main__":
    main()
