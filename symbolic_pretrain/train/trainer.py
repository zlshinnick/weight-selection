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
            model: The main model (e.g., TokenViTFixedPos wrapper) or list of models for parallel training.
            mlm_head: Masked language modeling head (linear layer) or list of heads for parallel training.
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

        # Check if we have parallel models
        self.is_parallel = isinstance(model, list)

        if self.is_parallel:
            self.n_models = len(model)
            # Move all models to device
            for m in self.model:
                m.to(self.device)
            for h in self.mlm_head:
                h.to(self.device)
            # Collect all unique parameters (deduplicate by id to handle tied weights)
            seen_param_ids = set()
            unique_params = []
            for m in self.model:
                for p in m.parameters():
                    if p.requires_grad and id(p) not in seen_param_ids:
                        seen_param_ids.add(id(p))
                        unique_params.append(p)
            for h in self.mlm_head:
                for p in h.parameters():
                    if p.requires_grad and id(p) not in seen_param_ids:
                        seen_param_ids.add(id(p))
                        unique_params.append(p)
            params = [{"params": unique_params}]
        else:
            self.n_models = 1
            self.model.to(self.device)
            self.mlm_head.to(self.device)
            params = [
                {"params": [p for p in self.model.parameters() if p.requires_grad]},
                {"params": self.mlm_head.parameters()},
            ]

        self.loader = DataLoader(
            dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
            pin_memory=cfg.dataset.pin_memory,
        )
        self.opt = make_optimizer(cfg, params)
        self.sched = make_scheduler(cfg, self.opt)
        self.ckpts = CheckpointManager(cfg)

    def step_once(self, batch):
        """Perform a single training step on a batch.

        Applies masking, forward pass, loss computation, backpropagation,
        gradient clipping (if enabled), and optimizer/scheduler updates.

        For parallel models:
        - Zero gradients
        - For each model i: forward & backward pass (accumulates gradients)
        - Divide shared parameter gradients by n
        - Update all models with the accumulated gradients

        Args:
            batch: Tensor of token IDs with shape (B, N).

        Returns:
            A tuple of (loss, accuracy, learning_rate, individual_accs) where:
                - loss: Scalar loss value for masked tokens (average for parallel models)
                - accuracy: Fraction of correctly predicted masked tokens (average for parallel models)
                - learning_rate: Current learning rate
                - individual_accs: List of accuracies for each model (only for parallel models, None otherwise)
            Returns (None, None, None, None) if no tokens were masked in this batch.
        """
        batch = batch.to(self.device)
        corrupt, target, mask = self.masking(batch)

        if self.is_parallel:
            # Parallel model training
            self.opt.zero_grad(set_to_none=True)

            losses = []
            accs = []

            # Forward and backward pass for each model
            for i in range(self.n_models):
                feats = self.model[i].forward_tokens(corrupt)
                logits = self.mlm_head[i](feats)
                masked_logits = logits[mask]
                masked_targets = target[mask]

                if masked_targets.numel() == 0:
                    continue

                loss = F.cross_entropy(masked_logits, masked_targets, reduction="mean")
                loss.backward()  # Accumulates gradients

                with torch.no_grad():
                    acc = (
                        (masked_logits.argmax(-1) == masked_targets)
                        .float()
                        .mean()
                        .item()
                    )

                losses.append(loss.item())
                accs.append(acc)

            if len(losses) == 0:
                return None, None, None, None

            # Divide gradients of shared parameters by n
            # Since tied parameters share the same tensor object, their gradients
            # have been accumulated n times (once per backward pass).
            # We need to identify which parameters are shared and divide their gradients by n.
            if self.n_models > 1:
                # Count how many times each parameter appears across all models
                # (shared/tied parameters will have count > 1)
                param_counts = {}
                for model_idx in range(self.n_models):
                    for p in self.model[model_idx].parameters():
                        if p.requires_grad:
                            param_id = id(p)
                            param_counts[param_id] = param_counts.get(param_id, 0) + 1
                    for p in self.mlm_head[model_idx].parameters():
                        if p.requires_grad:
                            param_id = id(p)
                            param_counts[param_id] = param_counts.get(param_id, 0) + 1

                # Collect unique parameters and divide gradients by their count
                seen_params = set()
                for model_idx in range(self.n_models):
                    for p in self.model[model_idx].parameters():
                        if p.requires_grad:
                            param_id = id(p)
                            if param_id not in seen_params:
                                seen_params.add(param_id)
                                if p.grad is not None and param_counts[param_id] > 1:
                                    # This is a shared parameter (appears in multiple models)
                                    # Divide by n since gradients accumulated n times
                                    p.grad.div_(self.n_models)
                    for p in self.mlm_head[model_idx].parameters():
                        if p.requires_grad:
                            param_id = id(p)
                            if param_id not in seen_params:
                                seen_params.add(param_id)
                                if p.grad is not None and param_counts[param_id] > 1:
                                    # This is a shared parameter
                                    p.grad.div_(self.n_models)

            # Gradient clipping
            if self.cfg.grad_clip.enabled:
                all_params = []
                for m in self.model:
                    all_params.extend([p for p in m.parameters() if p.requires_grad])
                for h in self.mlm_head:
                    all_params.extend(h.parameters())
                torch.nn.utils.clip_grad_norm_(
                    all_params,
                    self.cfg.grad_clip.max_norm,
                )

            self.opt.step()
            if self.sched is not None:
                self.sched.step()

            avg_loss = sum(losses) / len(losses)
            avg_acc = sum(accs) / len(accs) if accs else 0.0
            return avg_loss, avg_acc, self.opt.param_groups[0]["lr"], accs
        else:
            # Single model training (original behavior)
            feats = self.model.forward_tokens(corrupt)
            logits = self.mlm_head(feats)
            masked_logits = logits[mask]
            masked_targets = target[mask]
            if masked_targets.numel() == 0:
                return None, None, None, None
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
            return loss.item(), acc, self.opt.param_groups[0]["lr"], None

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
            if res == (None, None, None, None):
                continue
            loss, acc, lr, individual_accs = res
            if step % self.cfg.logging.print_freq == 0:
                if self.logger:
                    self.logger.log(step, loss, acc, lr)
                else:
                    print(
                        f"step {step:05d} | mlm_loss {loss:.4f} | mask_acc {acc:.3f} | lr {lr:.2e}"
                    )
            
            # Log individual model accuracies every 1000 steps for parallel models
            if self.is_parallel and individual_accs is not None and step % 1000 == 0:
                acc_str = " | ".join([f"model_{i}: {acc:.3f}" for i, acc in enumerate(individual_accs)])
                print(f"step {step:05d} | Individual model accuracies: {acc_str}")
                if self.logger:
                    # Log individual accuracies to wandb
                    extra_metrics = {
                        f"train/accuracy_model_{i}": acc for i, acc in enumerate(individual_accs)
                    }
                    self.logger.log(step, loss, acc, lr, extra=extra_metrics)
            # Save checkpoint
            if self.is_parallel:
                checkpoint_state = {
                    "step": step,
                    "n_models": self.n_models,
                    "model_states": [m.state_dict() for m in self.model],
                    "mlm_head_states": [h.state_dict() for h in self.mlm_head],
                    "optimizer_state": self.opt.state_dict(),
                    "scheduler_state": self.sched.state_dict() if self.sched else None,
                    "config": self.cfg,
                }
            else:
                checkpoint_state = {
                    "step": step,
                    "model_state": self.model.state_dict(),
                    "mlm_head_state": self.mlm_head.state_dict(),
                    "optimizer_state": self.opt.state_dict(),
                    "scheduler_state": self.sched.state_dict() if self.sched else None,
                    "config": self.cfg,
                }
            self.ckpts.maybe_save(step, checkpoint_state, logger=self.logger)
