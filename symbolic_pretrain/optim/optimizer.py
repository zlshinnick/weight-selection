"""Factory function for creating optimizers."""

from typing import Any, Iterator
import torch
import torch.nn as nn


def make_optimizer(cfg, params):
    """Create an AdamW optimizer with configuration settings.

    Args:
        cfg: Configuration object containing:
            - optimizer.lr: Learning rate
            - optimizer.weight_decay: Weight decay coefficient
            - optimizer.betas: Coefficients for computing running averages
        params: Model parameters to optimize (typically model.parameters()).

    Returns:
        A torch.optim.AdamW optimizer instance configured with the specified
        learning rate, weight decay, and betas.
    """
    return torch.optim.AdamW(
        params,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(cfg.optimizer.betas),
    )
