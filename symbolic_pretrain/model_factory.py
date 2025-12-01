"""Factory functions for building model components."""

import sys
from pathlib import Path
import torch.nn as nn

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
import utils

from .wrapper import TokenViTFixedPos
from .wrapper_swin import TokenSwinFixedPos
from .wrapper_convnext import TokenConvNeXtFixedPos


class _ArgsAdapter:
    """Adapter to convert cfg structure to args structure for utils.build_model."""

    def __init__(self, cfg):
        self.model = cfg.model.name
        self.nb_classes = cfg.model.num_classes
        self.drop_path = cfg.model.drop_path
        self.layer_scale_init_value = cfg.model.layer_scale_init_value
        self.head_init_scale = cfg.model.head_init_scale
        # For ConvNeXt, set in_chans to token embed dim so wrapper can feed BCHW
        if str(cfg.model.name).startswith("convnext"):
            self.in_chans = cfg.model.embed_dim
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
    backbone = make_backbone(cfg)

    use_swin = bool(getattr(cfg.model, "use_swin", False))
    if str(cfg.model.name).startswith("convnext"):
        # Expect ConvNeXt-like structure
        if not (hasattr(backbone, "num_features")):
            raise ValueError(
                "cfg.model.name startswith 'convnext' but backbone missing num_features."
            )
        wrap = TokenConvNeXtFixedPos(
            convnext_backbone=backbone,
            tok_embed=tok_embed,
            H=cfg.grid.H,
            W=cfg.grid.W,
            pos_embed_2d=pos_embed,
            return_tokens=True,
        )
        feat_dim = getattr(backbone, "num_features")
        mlm = make_mlm_head(feat_dim, cfg.vocab.K)
        return wrap, mlm
    elif use_swin:
        # Basic sanity: expect Swin-like structure
        if not hasattr(backbone, "layers") or not hasattr(backbone, "norm"):
            raise ValueError(
                "cfg.model.use_swin=True but cfg.model.name is not a Swin model. "
                "Set cfg.model.name to a timm Swin variant (e.g., 'swin_tiny_patch4_window7_224')."
            )
        wrap = TokenSwinFixedPos(
            swin_backbone=backbone,
            tok_embed=tok_embed,
            H=cfg.grid.H,
            W=cfg.grid.W,
            pos_embed_2d=pos_embed,
        )
        feat_dim = getattr(backbone, "num_features", getattr(backbone, "embed_dim"))
        mlm = make_mlm_head(feat_dim, cfg.vocab.K)
        return wrap, mlm
    else:
        vit = backbone
        wrap = TokenViTFixedPos(
            vit_backbone=vit,
            tok_embed=tok_embed,
            pos_embed=pos_embed,
            H=cfg.grid.H,
            W=cfg.grid.W,
        )
        mlm = make_mlm_head(cfg.model.embed_dim, cfg.vocab.K)
        return wrap, mlm
