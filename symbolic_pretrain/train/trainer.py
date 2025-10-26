"""Training loop and checkpoint management for masked language modeling."""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..optim.optimizer import make_optimizer
from ..optim.schedulers import make_scheduler


class CheckpointManager:
    """Manages saving model checkpoints at specified training steps.

    This class handles creating the checkpoint directory and saving model
    state dictionaries at configured intervals during training.
    """

    def __init__(self, cfg):
        """Initialize the checkpoint manager.

        Args:
            cfg: Configuration object containing:
                - checkpoint.dir: Directory to save checkpoints
                - checkpoint.save_steps: List of steps at which to save checkpoints
        """
        self.dir = cfg.checkpoint.dir
        self.steps = set(cfg.checkpoint.save_steps)
        os.makedirs(self.dir, exist_ok=True)

    def path(self, step):
        """Generate the checkpoint file path for a given step.

        Args:
            step: Training step number.

        Returns:
            Path string for the checkpoint file.
        """
        return os.path.join(self.dir, f"ckpt_step_{step:06d}.pt")

    def maybe_save(self, step, state, logger=None):
        """Save a checkpoint if the current step is in the save list.

        Args:
            step: Current training step number.
            state: Dictionary containing model state to save.
            logger: Optional logger instance to upload checkpoint as artifact.
        """
        if step in self.steps:
            p = self.path(step)
            torch.save(state, p)
            print(f"[checkpoint] saved -> {p}")
            if logger and hasattr(logger, "add_artifact"):
                logger.add_artifact(p, f"ckpt-{step}")


class Trainer:
    """Trainer for masked language modeling on symbolic sequences.

    This class orchestrates the training loop, including data loading,
    forward/backward passes, optimization, logging, and checkpointing.
    """

    def __init__(self, cfg, model, mlm_head, dataset, masking, logger=None):
        """Initialize the trainer with model, dataset, and configuration.

        Args:
            cfg: Configuration object containing training settings:
                - training.device: Device to use ('cuda' or 'cpu')
                - training.steps: Total number of training steps
                - dataset.batch_size: Batch size for training
                - dataset.num_workers: Number of data loading workers
                - dataset.pin_memory: Whether to pin memory for data loading
                - grad_clip.enabled: Whether to clip gradients
                - grad_clip.max_norm: Maximum gradient norm if clipping enabled
                - logging.print_freq: Frequency of logging in steps
            model: The main model (e.g., TokenViTFixedPos wrapper).
            mlm_head: Masked language modeling head (linear layer).
            dataset: PyTorch Dataset for generating training sequences.
            masking: Masking strategy instance (e.g., CloseOnlyMasking).
            logger: Optional logger instance for tracking metrics.
        """
        self.cfg, self.model, self.mlm_head, self.masking = (
            cfg,
            model,
            mlm_head,
            masking,
        )
        self.logger = logger
        self.device = cfg.training.device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.mlm_head.to(self.device)
        self.loader = DataLoader(
            dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
            pin_memory=cfg.dataset.pin_memory,
        )
        params = [
            {"params": [p for p in self.model.parameters() if p.requires_grad]},
            {"params": self.mlm_head.parameters()},
        ]
        self.opt = make_optimizer(cfg, params)
        self.sched = make_scheduler(cfg, self.opt)
        self.ckpts = CheckpointManager(cfg)

    def step_once(self, batch):
        """Perform a single training step on a batch.

        Applies masking, forward pass, loss computation, backpropagation,
        gradient clipping (if enabled), and optimizer/scheduler updates.

        Args:
            batch: Tensor of token IDs with shape (B, N).

        Returns:
            A tuple of (loss, accuracy, learning_rate) where:
                - loss: Scalar loss value for masked tokens
                - accuracy: Fraction of correctly predicted masked tokens
                - learning_rate: Current learning rate
            Returns (None, None, None) if no tokens were masked in this batch.
        """
        batch = batch.to(self.device)
        corrupt, target, mask = self.masking(batch)
        feats = self.model.forward_tokens(corrupt)
        logits = self.mlm_head(feats)
        masked_logits = logits[mask]
        masked_targets = target[mask]
        if masked_targets.numel() == 0:
            return None, None, None
        loss = F.cross_entropy(masked_logits, masked_targets, reduction="mean")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip.enabled:
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.mlm_head.parameters()),
                self.cfg.grad_clip.max_norm,
            )
        self.opt.step()
        if self.sched is not None:
            self.sched.step()
        with torch.no_grad():
            acc = (masked_logits.argmax(-1) == masked_targets).float().mean().item()
        return loss.item(), acc, self.opt.param_groups[0]["lr"]

    def train(self):
        """Execute the main training loop for the configured number of steps.

        Iterates through the data loader, performing training steps, logging
        metrics at specified intervals, and saving checkpoints at configured steps.
        The data loader is automatically restarted when exhausted to continue training
        for the full number of configured steps.
        """
        it = iter(self.loader)
        steps = self.cfg.training.steps
        for step in range(1, steps + 1):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.loader)
                batch = next(it)
            res = self.step_once(batch)
            if res == (None, None, None):
                continue
            loss, acc, lr = res
            if step % self.cfg.logging.print_freq == 0:
                if self.logger:
                    self.logger.log(step, loss, acc, lr)
                else:
                    print(
                        f"step {step:05d} | mlm_loss {loss:.4f} | mask_acc {acc:.3f} | lr {lr:.2e}"
                    )
            self.ckpts.maybe_save(
                step,
                {
                    "step": step,
                    "model_state": self.model.state_dict(),
                    "mlm_head_state": self.mlm_head.state_dict(),
                    "optimizer_state": self.opt.state_dict(),
                    "scheduler_state": self.sched.state_dict() if self.sched else None,
                    "config": self.cfg,
                },
                logger=self.logger,
            )
