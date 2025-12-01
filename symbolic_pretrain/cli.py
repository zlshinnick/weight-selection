"""Command-line interface for training symbolic pretraining models."""

import argparse
import os
import yaml
from dataclasses import asdict
from .cfg import load_cfg
from .utils import set_all_seeds
from .vocab import FrozenTokenEmbedding
from .pos_embed import FrozenPositionalEmbedding
from .model_factory import build_model
from .data.dyck.dataset import DyckGrid
from .data.dyck.masking import (
    CloseOnlyMasking as DyckCloseOnlyMasking,
    RandomMasking as DyckRandomMasking,
)
from .data.dyck_shuffle.dataset import DyckShuffleGrid
from .data.dyck_shuffle.masking import (
    CloseOnlyMasking as ShuffleCloseOnlyMasking,
    RandomMasking as ShuffleRandomMasking,
)
from .data.ww.dataset import WwGrid
from .data.ww.masking import (
    CloseOnlyMasking as WwCloseOnlyMasking,
    RandomMasking as WwRandomMasking,
)
from .train.logger import WandbLogger
from .train.trainer import Trainer


def main():
    """Main entry point for training a masked language model on Dyck sequences.

    This function orchestrates the complete training pipeline:
    1. Parses command-line arguments for configuration and masking strategy
    2. Loads configuration and sets random seeds for reproducibility
    3. Creates frozen token and positional embeddings
    4. Builds the ViT model and MLM head
    5. Initializes dataset and masking strategy
    6. Sets up optional W&B logging
    7. Runs the training loop
    8. Cleans up resources

    Command-line arguments:
        --config: Path to YAML configuration file (default: None)
        --masking: Masking strategy to use, either 'close_only' or 'random'
                  (default: 'close_only')
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument(
        "--masking", type=str, default="close_only", choices=["close_only", "random"]
    )
    ap.add_argument(
        "--use-swin",
        action="store_true",
        help="Use Swin wrapper instead of ViT (ablation)",
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    # CLI override for ablation toggle
    if getattr(args, "use_swin", False):
        cfg.model.use_swin = True
    set_all_seeds(cfg.seed)

    # Save config to checkpoint directory
    os.makedirs(cfg.checkpoint.dir, exist_ok=True)
    config_save_path = os.path.join(cfg.checkpoint.dir, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False, sort_keys=False)
    print(f"[config] saved to {config_save_path}")

    # embeddings
    N = cfg.grid.H * cfg.grid.W
    tok = FrozenTokenEmbedding(cfg.vocab.K, cfg.model.embed_dim)
    pos = FrozenPositionalEmbedding(N, cfg.model.embed_dim)

    # model + head
    model, mlm_head = build_model(cfg, tok, pos)

    # data + masking
    source = getattr(cfg.dataset, "source", "dyck")
    if source == "dyck_shuffle":
        ds = DyckShuffleGrid(cfg)
        masking = (
            ShuffleCloseOnlyMasking(cfg)
            if args.masking == "close_only"
            else ShuffleRandomMasking(cfg)
        )
    elif source == "ww":
        ds = WwGrid(cfg)
        # For ww, "close_only" maps to the ww-specific masking policy
        masking = (
            WwCloseOnlyMasking(cfg)
            if args.masking == "close_only"
            else WwRandomMasking(cfg)
        )
    else:
        ds = DyckGrid(cfg)
        masking = (
            DyckCloseOnlyMasking(cfg)
            if args.masking == "close_only"
            else DyckRandomMasking(cfg)
        )

    # logger
    logger = None
    if cfg.wandb.enabled:
        try:
            logger = WandbLogger(cfg)
            logger.watch(model, cfg.logging.print_freq)
        except Exception as e:
            print(f"[wandb] failed to init: {e}")

    # train
    trainer = Trainer(cfg, model, mlm_head, ds, masking, logger=logger)
    trainer.train()
    if logger and hasattr(logger, "finish"):
        logger.finish()


if __name__ == "__main__":
    main()
