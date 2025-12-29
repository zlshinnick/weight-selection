"""Command-line interface for training symbolic pretraining models."""

import argparse
import os
import yaml
from dataclasses import asdict
from .cfg import load_cfg
from .utils import set_all_seeds
from .vocab import FrozenTokenEmbedding
from .pos_embed import FrozenPositionalEmbedding
from .model_factory import build_model, build_parallel_models
from .apply_low_rank import apply_low_rank_factorization
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

    # Read num_parallel_models from config
    num_parallel_models = getattr(cfg.model, "num_parallel_models", 1)
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
    if num_parallel_models > 1:
        print(f"\n[model] Building {num_parallel_models} parallel models...")
        models, mlm_heads = build_parallel_models(cfg, tok, pos, num_parallel_models)
        model = models  # Trainer will handle list of models
        mlm_head = mlm_heads  # Trainer will handle list of heads

        # Log parallel model details
        print(f"[model] ✓ Created {len(models)} parallel models")
        print(f"[model] Architecture: {cfg.model.name}")
        print(f"[model] Embed dim: {cfg.model.embed_dim}")
        print(
            f"[model] Weight tying within models: {getattr(cfg.model, 'tie_weights', False)}"
        )
        print("[model] Weight tying across models: enabled (attention & MLP)")

        # Calculate parameter counts
        total_params = sum(p.numel() for m in models for p in m.parameters())
        total_trainable = sum(
            p.numel() for m in models for p in m.parameters() if p.requires_grad
        )
        mlm_params = sum(p.numel() for h in mlm_heads for p in h.parameters())
        mlm_trainable = sum(
            p.numel() for h in mlm_heads for p in h.parameters() if p.requires_grad
        )

        # Count unique parameters (accounting for weight tying)
        seen_param_ids = set()
        unique_params = 0
        unique_trainable = 0
        for m in models:
            for p in m.parameters():
                param_id = id(p)
                if param_id not in seen_param_ids:
                    seen_param_ids.add(param_id)
                    unique_params += p.numel()
                    if p.requires_grad:
                        unique_trainable += p.numel()
        for h in mlm_heads:
            for p in h.parameters():
                param_id = id(p)
                if param_id not in seen_param_ids:
                    seen_param_ids.add(param_id)
                    unique_params += p.numel()
                    if p.requires_grad:
                        unique_trainable += p.numel()

        print("[model] Parameter counts:")
        print(f"  - Total params (all models): {total_params:,}")
        print(f"  - Unique params (after tying): {unique_params:,}")
        print(f"  - Trainable params (all models): {total_trainable:,}")
        print(f"  - Unique trainable (after tying): {unique_trainable:,}")
        print(f"  - MLM head params (all heads): {mlm_params:,}")
        print(f"  - MLM head trainable: {mlm_trainable:,}")

        # Show seeds used for each model
        print(
            f"[model] Model seeds: {[cfg.seed + i for i in range(num_parallel_models)]}"
        )
        print("[model] Models initialized and weights tied across models\n")
    else:
        model, mlm_head = build_model(cfg, tok, pos)
        print(f"[model] Built single model: {cfg.model.name}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[model] Total params: {total_params:,}, Trainable: {trainable_params:,}\n"
        )

    # Apply low-rank factorization if requested
    using_low_rank = getattr(cfg.model, "rank_attn", None) is not None or getattr(cfg.model, "rank_mlp", None) is not None
    if using_low_rank:
        apply_low_rank_factorization(model, cfg, total_params)

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
            # Watch first model if parallel, or single model
            model_to_watch = model[0] if isinstance(model, list) else model
            logger.watch(model_to_watch, cfg.logging.print_freq)
        except Exception as e:
            print(f"[wandb] failed to init: {e}")

    # train
    trainer = Trainer(cfg, model, mlm_head, ds, masking, logger=logger)
    trainer.train()
    if logger and hasattr(logger, "finish"):
        logger.finish()


if __name__ == "__main__":
    main()
