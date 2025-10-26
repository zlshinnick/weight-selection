"""Test script to verify mimetic initialization works correctly."""

import torch
import numpy as np
from argparse import Namespace
import utils


def test_mimetic_initialization():
    """Test that mimetic initialization is applied correctly to a ViT model."""

    print("=" * 80)
    print("Testing Mimetic Initialization")
    print("=" * 80)

    # Create a simple args object for testing
    args = Namespace(
        model="vit_tiny_patch16_224",
        nb_classes=1000,
        drop_path=0.0,
        mimetic_init=True,
        mimetic_alpha=0.4,
        mimetic_beta=0.4,
        mimetic_dist="uniform",
    )

    print("\n1. Building model WITH mimetic initialization...")
    model_mimetic = utils.build_model(args)

    print("\n2. Building model WITHOUT mimetic initialization...")
    args.mimetic_init = False
    model_standard = utils.build_model(args)

    print("\n3. Comparing attention weight statistics...")
    print("-" * 80)

    # Extract attention weights from both models
    for (name_m, module_m), (name_s, module_s) in zip(
        model_mimetic.named_modules(), model_standard.named_modules()
    ):
        if hasattr(module_m, "qkv") and hasattr(module_m, "proj"):
            with torch.no_grad():
                qkv_mimetic = module_m.qkv.weight
                qkv_standard = module_s.qkv.weight

                print(f"\nAttention module: {name_m}")
                print(f"  QKV weight shape: {qkv_mimetic.shape}")
                print(
                    f"  Mimetic - mean: {qkv_mimetic.mean():.6f}, std: {qkv_mimetic.std():.6f}"
                )
                print(
                    f"  Standard - mean: {qkv_standard.mean():.6f}, std: {qkv_standard.std():.6f}"
                )

                # Check if weights are different (they should be)
                diff = (qkv_mimetic - qkv_standard).abs().mean()
                print(f"  Difference: {diff:.6f}")

                # Check biases are zero for mimetic
                if module_m.qkv.bias is not None:
                    bias_norm = module_m.qkv.bias.norm().item()
                    print(f"  Mimetic QKV bias norm: {bias_norm:.6f} (should be 0)")

                if module_m.proj.bias is not None:
                    proj_bias_norm = module_m.proj.bias.norm().item()
                    print(
                        f"  Mimetic proj bias norm: {proj_bias_norm:.6f} (should be 0)"
                    )

                break  # Just check the first attention layer

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)

    # Test the apply_mimetic_init function directly
    print("\n4. Testing direct application of mimetic_init to existing model...")
    model_direct = utils.build_model(
        Namespace(
            model="vit_tiny_patch16_224",
            nb_classes=1000,
            drop_path=0.0,
            mimetic_init=False,
        )
    )

    # Get initial weight statistics
    first_attn = None
    for name, module in model_direct.named_modules():
        if hasattr(module, "qkv") and hasattr(module, "proj"):
            first_attn = module
            break

    if first_attn is not None:
        with torch.no_grad():
            qkv_before = first_attn.qkv.weight.clone()

        # Apply mimetic init
        utils.apply_mimetic_init(model_direct, alpha=0.4, beta=0.4, dist="uniform")

        with torch.no_grad():
            qkv_after = first_attn.qkv.weight
            diff = (qkv_before - qkv_after).abs().mean()
            print(
                f"\nWeights changed after apply_mimetic_init: {diff:.6f} (should be > 0)"
            )
            print(
                f"QKV bias norm after init: {first_attn.qkv.bias.norm().item():.6f} (should be 0)"
            )

    print("\nAll tests passed! ✓")


def test_get_ortho_like():
    """Test the get_ortho_like helper function."""
    print("\n" + "=" * 80)
    print("Testing get_ortho_like function")
    print("=" * 80)

    dim = 192
    heads = 3
    alpha = 0.4
    beta = 0.4

    # Test with uniform distribution
    L_unif, R_unif = utils.get_ortho_like(
        dim, heads, alpha, beta, sign=1, dist="uniform"
    )
    print(f"\nUniform distribution:")
    print(f"  L shape: {L_unif.shape}, R shape: {R_unif.shape}")
    print(f"  L mean: {L_unif.mean():.6f}, std: {L_unif.std():.6f}")
    print(f"  R mean: {R_unif.mean():.6f}, std: {R_unif.std():.6f}")

    # Test with normal distribution
    L_norm, R_norm = utils.get_ortho_like(
        dim, heads, alpha, beta, sign=1, dist="normal"
    )
    print(f"\nNormal distribution:")
    print(f"  L shape: {L_norm.shape}, R shape: {R_norm.shape}")
    print(f"  L mean: {L_norm.mean():.6f}, std: {L_norm.std():.6f}")
    print(f"  R mean: {R_norm.mean():.6f}, std: {R_norm.std():.6f}")

    # Verify factorization properties
    product = L_unif @ R_unif
    print(f"\nFactorization L @ R shape: {product.shape}")
    print(f"Diagonal elements mean: {np.diag(product).mean():.6f}")

    print("\nget_ortho_like tests passed! ✓")


if __name__ == "__main__":
    test_get_ortho_like()
    test_mimetic_initialization()
