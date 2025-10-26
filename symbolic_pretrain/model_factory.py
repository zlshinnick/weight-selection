"""Factory functions for building model components."""

import sys
from pathlib import Path
import torch.nn as nn

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
import utils

from .wrapper import TokenViTFixedPos


class _ArgsAdapter:
    """Adapter to convert cfg structure to args structure for utils.build_model."""

    def __init__(self, cfg):
        self.model = cfg.model.name
        self.nb_classes = cfg.model.num_classes
        self.drop_path = cfg.model.drop_path
        self.layer_scale_init_value = cfg.model.layer_scale_init_value
        self.head_init_scale = cfg.model.head_init_scale
        # Pass mimetic init flags through to utils.build_model
        self.mimetic_init = getattr(cfg.model, "mimetic_init", False)
        self.mimetic_alpha = getattr(cfg.model, "mimetic_alpha", 0.4)
        self.mimetic_beta = getattr(cfg.model, "mimetic_beta", 0.4)
        self.mimetic_dist = getattr(cfg.model, "mimetic_dist", "uniform")


def make_backbone(cfg):
    """Create a Vision Transformer backbone using utils.build_model.

    Args:
        cfg: Configuration object containing model settings.

    Returns:
        A model instance created via utils.build_model.
        Model is always initialized from scratch (pretrained=False) to match main.py behavior.
    """
    args_adapter = _ArgsAdapter(cfg)
    return utils.build_model(args_adapter)


def make_mlm_head(embed_dim: int, K: int):
    """Create a masked language modeling head.

    Args:
        embed_dim: Dimension of the input embeddings.
        K: Vocabulary size (number of output classes).

    Returns:
        A linear layer mapping from embed_dim to K classes.
    """
    return nn.Linear(embed_dim, K)


def build_model(cfg, tok_embed: nn.Module, pos_embed: nn.Module):
    """Build the complete model with backbone, embeddings, and MLM head.

    Args:
        cfg: Configuration object containing model, grid, and vocab settings.
        tok_embed: Token embedding module.
        pos_embed: Positional embedding module.

    Returns:
        A tuple of (wrapped_model, mlm_head) where:
            - wrapped_model: TokenViTFixedPos wrapper around the ViT backbone.
            - mlm_head: Linear layer for masked language modeling predictions.
    """
    vit = make_backbone(cfg)
    wrap = TokenViTFixedPos(
        vit_backbone=vit,
        tok_embed=tok_embed,
        pos_embed=pos_embed,
        H=cfg.grid.H,
        W=cfg.grid.W,
    )
    mlm = make_mlm_head(cfg.model.embed_dim, cfg.vocab.K)
    return wrap, mlm
