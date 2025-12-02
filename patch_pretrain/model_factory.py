"""Factory helpers for patch_pretrain models."""

import sys
from pathlib import Path
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
import utils  # noqa: E402


class _ArgsAdapter:
    def __init__(self, cfg):
        self.model = cfg.model.name
        self.nb_classes = cfg.model.num_classes
        self.drop_path = cfg.model.drop_path
        self.layer_scale_init_value = cfg.model.layer_scale_init_value
        self.head_init_scale = cfg.model.head_init_scale
        self.in_chans = cfg.model.in_chans
        self.img_size = cfg.model.img_size
        self.patch_size = cfg.model.patch_size
        self.pretrained = cfg.model.pretrained


def make_backbone(cfg):
    adapter = _ArgsAdapter(cfg)
    return utils.build_model(adapter)


def build_model(cfg):
    backbone = make_backbone(cfg)
    feat_dim = getattr(backbone, "num_features", getattr(backbone, "embed_dim"))
    mlm_head = nn.Linear(feat_dim, cfg.vocab.K)
    return backbone, mlm_head

