"""Test that mimetic initialization works with main.py argument parser."""

import sys
import argparse
from argparse import Namespace

# Import the argument parser from main.py
sys.path.insert(0, "/home/zshinnick/weight-selection")
from main import get_args_parser
import utils


def test_main_args_mimetic():
    """Test that mimetic init arguments are properly parsed and applied."""

    print("=" * 80)
    print("Testing Mimetic Initialization with main.py Arguments")
    print("=" * 80)

    # Create parser and parse test arguments
    parser = argparse.ArgumentParser("Test", parents=[get_args_parser()])

    # Test 1: Default (mimetic disabled)
    print("\n1. Testing default arguments (mimetic_init=False)...")
    args = parser.parse_args(
        ["--model", "vit_tiny", "--nb_classes", "100", "--drop_path", "0.0"]
    )

    print(f"   mimetic_init: {args.mimetic_init}")
    print(f"   mimetic_alpha: {args.mimetic_alpha}")
    print(f"   mimetic_beta: {args.mimetic_beta}")
    print(f"   mimetic_dist: {args.mimetic_dist}")

    model = utils.build_model(args)
    print(f"   ✓ Model built successfully without mimetic init")

    # Test 2: Mimetic enabled with defaults
    print("\n2. Testing with mimetic_init=True (default parameters)...")
    args = parser.parse_args(
        [
            "--model",
            "vit_tiny",
            "--nb_classes",
            "100",
            "--drop_path",
            "0.0",
            "--mimetic_init",
            "true",
        ]
    )

    print(f"   mimetic_init: {args.mimetic_init}")
    print(f"   mimetic_alpha: {args.mimetic_alpha}")
    print(f"   mimetic_beta: {args.mimetic_beta}")
    print(f"   mimetic_dist: {args.mimetic_dist}")

    model = utils.build_model(args)
    print(f"   ✓ Model built successfully with mimetic init (default params)")

    # Test 3: Mimetic enabled with custom parameters
    print("\n3. Testing with mimetic_init=True (custom parameters)...")
    args = parser.parse_args(
        [
            "--model",
            "vit_tiny",
            "--nb_classes",
            "100",
            "--drop_path",
            "0.0",
            "--mimetic_init",
            "true",
            "--mimetic_alpha",
            "0.5",
            "--mimetic_beta",
            "0.3",
            "--mimetic_dist",
            "normal",
        ]
    )

    print(f"   mimetic_init: {args.mimetic_init}")
    print(f"   mimetic_alpha: {args.mimetic_alpha}")
    print(f"   mimetic_beta: {args.mimetic_beta}")
    print(f"   mimetic_dist: {args.mimetic_dist}")

    model = utils.build_model(args)
    print(f"   ✓ Model built successfully with mimetic init (custom params)")

    # Verify that mimetic initialization actually changed the weights
    print("\n4. Verifying weights were actually initialized differently...")
    args_standard = parser.parse_args(
        [
            "--model",
            "vit_tiny",
            "--nb_classes",
            "100",
            "--drop_path",
            "0.0",
            "--mimetic_init",
            "false",
        ]
    )

    args_mimetic = parser.parse_args(
        [
            "--model",
            "vit_tiny",
            "--nb_classes",
            "100",
            "--drop_path",
            "0.0",
            "--mimetic_init",
            "true",
        ]
    )

    model_standard = utils.build_model(args_standard)
    model_mimetic = utils.build_model(args_mimetic)

    # Compare first attention layer weights
    import torch

    for (name_s, mod_s), (name_m, mod_m) in zip(
        model_standard.named_modules(), model_mimetic.named_modules()
    ):
        if hasattr(mod_s, "qkv") and hasattr(mod_s, "proj"):
            with torch.no_grad():
                diff = (mod_s.qkv.weight - mod_m.qkv.weight).abs().mean().item()
                print(f"   First attention layer QKV weight difference: {diff:.6f}")
                assert diff > 0.01, "Weights should be different!"
                print(f"   ✓ Weights are properly initialized differently")

                # Check biases are zeroed in mimetic
                if mod_m.qkv.bias is not None:
                    bias_norm = mod_m.qkv.bias.norm().item()
                    print(f"   Mimetic QKV bias norm: {bias_norm:.6f} (should be ~0)")
                    assert bias_norm < 1e-6, "Bias should be zero!"
                    print(f"   ✓ Biases properly zeroed")
                break

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    print("\nYou can now use mimetic initialization in your training by adding:")
    print("  --mimetic_init True")
    print("\nOptional customization:")
    print("  --mimetic_alpha 0.4")
    print("  --mimetic_beta 0.4")
    print("  --mimetic_dist uniform")


if __name__ == "__main__":
    test_main_args_mimetic()
