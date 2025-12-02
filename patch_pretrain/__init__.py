"""
Patch-based pretraining pipeline that operates directly on image-like inputs.

This package mirrors the high-level structure of `symbolic_pretrain` but avoids
the token/position wrapper so we can feed ViT-style backbones with tiled Gabor
filters (or other patch-level signals).
"""
