# Patch Pretraining

This module mirrors the symbolic pretraining pipeline but feeds standard ViT
backbones with tiled Gabor filters instead of relying on the token/position
wrapper. Each Dyck token is mapped to a padded Gabor kernel and the dataset
tiles those kernels into a single-channel image so ViT patch embeddings can be
used directly.

## Quick Start

```bash
python -m patch_pretrain.cli \
  --config patch_pretrain/configs/dyck_gabor_patch.yaml \
  --masking close_only
```

The default config yields:

- `grid.H = grid.W = 8` (64 tokens per sample)
- `n_freqs = 8`, `n_thetas = 16` → 128 unique Gabor filters
- Image size of `136 × 136` with patch size `17`, so each ViT patch lines up
  with exactly one Dyck token.

You can edit the YAML to change ViT variants (e.g., `vit_small_patch16_224`),
Gabor parameters, or training hyperparameters. When new parameters require a
different patch size, the CLI automatically updates `cfg.model.patch_size` and
`cfg.model.img_size` to keep the ViT configuration consistent with the dataset.

