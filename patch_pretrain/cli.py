"""Patch-based CLI that feeds standard ViT backbones with tiled Gabor images."""

import argparse
import os
import yaml
from dataclasses import asdict

from symbolic_pretrain.train.logger import WandbLogger

from .cfg import load_cfg, cfg_to_dict
from .utils import set_all_seeds
from .data.dyck_gabor.dataset import DyckGaborImageDataset
from .data.dyck_gabor.masking import CloseOnlyMasking, RandomMasking
from .model_factory import build_model
from .trainer import Trainer


def _save_config(cfg):
    os.makedirs(cfg.checkpoint.dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint.dir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False, sort_keys=False)
    print(f"[config] saved to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument(
        "--masking", type=str, default="close_only", choices=["close_only", "random"]
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_all_seeds(cfg.seed)

    dataset = DyckGaborImageDataset(cfg)
    cfg.model.patch_size = dataset.patch_h
    cfg.model.img_size = dataset.image_height
    cfg.model.in_chans = dataset.channels

    model, mlm_head = build_model(cfg)

    masking = (
        CloseOnlyMasking(cfg)
        if args.masking == "close_only"
        else RandomMasking(cfg)
    )

    logger = None
    if cfg.wandb.enabled:
        try:
            logger = WandbLogger(cfg)
            logger.watch(model, cfg.logging.print_freq)
        except Exception as exc:
            print(f"[wandb] failed to init: {exc}")

    trainer = Trainer(cfg, model, mlm_head, dataset, masking, logger=logger)
    _save_config(cfg)
    trainer.train()
    if logger and hasattr(logger, "finish"):
        logger.finish()


if __name__ == "__main__":
    main()

