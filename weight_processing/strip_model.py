#!/usr/bin/env python3
"""
Convert and clean pretrain checkpoints from .pt or .pth format.
Removes MLM-specific weights and untrained components.
"""

import sys
from pathlib import Path

# Add parent directory to path so symbolic_pretrain can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def convert_and_clean_checkpoint(input_path, output_path):
    """
    Convert checkpoint (.pt or .pth) to clean .pth format.

    Steps:
    1. Load the checkpoint
    2. Extract model_state (handles 'model_state', 'model', 'state_dict', or raw state_dict)
    3. Strip wrapper prefixes (vit., tok., pos.)
    4. Remove untrained components (patch_embed, pos_embed, head)
    5. Save as clean .pth file
    """
    print(f"\nProcessing: {input_path.name}")
    print("-" * 80)

    # Load checkpoint
    print("  Loading checkpoint...")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    if hasattr(checkpoint, "keys"):
        print(f"  Checkpoint keys: {list(checkpoint.keys())}")
    else:
        print(f"  Checkpoint type: {type(checkpoint)}")

    # Extract model state
    # Support multiple common layouts
    if isinstance(checkpoint, dict):
        # Handle parallel model checkpoints (model_states is a list)
        if "model_states" in checkpoint:
            model_states = checkpoint["model_states"]
            if isinstance(model_states, list) and len(model_states) > 0:
                # Use the first model as reference (all models share tied weights)
                n_models = checkpoint.get("n_models", len(model_states))
                print(f"  Parallel model checkpoint detected: {n_models} models")
                print(f"  Using reference model (model[0]) - all models share tied weights")
                model_state_dict = model_states[0]
            else:
                raise ValueError("model_states is not a valid list")
        elif "model_state" in checkpoint:
            model_state_dict = checkpoint["model_state"]
        elif "model" in checkpoint:
            model_state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            model_state_dict = checkpoint["state_dict"]
        else:
            # If dict maps param_name -> tensor, treat as raw state_dict
            values = list(checkpoint.values())
            if values and all(hasattr(v, "shape") for v in values):
                model_state_dict = checkpoint
            else:
                raise ValueError(
                    f"Unknown checkpoint format keys: {list(checkpoint.keys())}"
                )
    elif hasattr(checkpoint, "items"):
        # OrderedDict-like raw state_dict
        model_state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint object; expected dict or state_dict")

    print(f"  Original model keys: {len(model_state_dict)}")

    # Clean the state dict
    cleaned_state_dict = {}
    removed_keys = []

    # Keys / prefixes to remove (untrained or MLM-specific)
    # - Handle both ViT-B and XCiT naming: prefer prefix-based removal
    skip_prefixes = (
        "patch_embed.",  # image patch embedding (unused/untrained in symbolic pretrain)
        "pos_embed.",  # backbone positional embed (unused/untrained)
        "head.",  # classifier head (unused/untrained)
    )
    skip_exact = {
        "pos_embed",  # some ViT variants store the whole tensor under this key
        "cls_token",  # class token (unused in our XCiT path)
        "tok.weight",  # MLM token embeddings (wrapper)
        "pos.weight",  # MLM positional embeddings (wrapper)
        "patch_embed.proj.weight",  # legacy exact names
        "patch_embed.proj.bias",
        "head.weight",
        "head.bias",
    }

    for key, value in model_state_dict.items():
        # Skip MLM-specific wrapper keys (if present at top-level)
        if key in ("tok.weight", "pos.weight"):
            removed_keys.append(f"{key} (MLM-specific)")
            continue

        # Normalize by stripping 'vit.' prefix when present
        if key.startswith("vit."):
            new_key = key[4:]  # Remove 'vit.' prefix
        else:
            new_key = key

        # Decide to skip based on exact or prefix match
        if new_key in skip_exact or any(new_key.startswith(p) for p in skip_prefixes):
            removed_keys.append(f"{key} -> {new_key} (untrained)")
            continue

        cleaned_state_dict[new_key] = value

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
    # Determine target (optional CLI arg: file or directory)
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if target.is_file():
            checkpoint_dir = target.parent
            input_files = [target]
        else:
            checkpoint_dir = target
            input_files = []
    else:
        # Default directory containing pretrain checkpoints
        checkpoint_dir = Path("fractal-pt-models/v7/")
        input_files = []

    # Find .pt and .pth files
    if not input_files:
        pt_files = list(checkpoint_dir.glob("ckpt_step_*.pt"))
        pth_files = [
            p
            for p in checkpoint_dir.glob("*.pth")
            if not p.name.endswith("_no_embed.pth")
        ]
        input_files = sorted(pt_files + pth_files)

    if not input_files:
        print(f"No checkpoint .pt or .pth files found in {checkpoint_dir}")
        return

    print("=" * 80)
    print("CONVERTING AND CLEANING PRETRAIN CHECKPOINTS")
    print("=" * 80)
    print(f"\nFound {len(input_files)} checkpoint(s) to process")
    print("\nOperations:")
    print("  1. Extract model weights from .pt checkpoint")
    print("  2. Strip 'vit.' prefix from transformer weights")
    print("  3. Remove MLM-specific weights (tok, pos embeddings)")
    print("  4. Remove untrained components (patch_embed, pos_embed, head)")
    print("  5. Save as clean .pth file for fine-tuning")

    results = {}
    failures = []
    for in_file in input_files:
        # Create output filename (normalize to *_no_embed.pth; avoid double-suffix)
        base_name = in_file.stem
        if base_name.endswith("_no_embed"):
            out_name = f"{base_name}_cleaned.pth"
        else:
            out_name = f"{base_name}_no_embed.pth"
        pth_file = checkpoint_dir / out_name

        try:
            num_keys, num_removed = convert_and_clean_checkpoint(in_file, pth_file)
            results[pth_file.name] = (num_keys, num_removed)
        except Exception as e:
            print(f"\n  ✗ Error processing {in_file.name}: {e}\n")
            failures.append((in_file.name, str(e)))

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
        print(
            f"\n⚠ {len(results)}/{len(input_files)} checkpoints processed successfully."
        )
        print(f"Cleaned checkpoints saved in: {checkpoint_dir}")
    else:
        print("\n✗ All checkpoints failed to process. Please check the errors above.")


if __name__ == "__main__":
    main()
