"""Training loop for the patch-based Gabor pretraining pipeline."""

from __future__ import annotations

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from symbolic_pretrain.optim.optimizer import make_optimizer
from symbolic_pretrain.optim.schedulers import make_scheduler


class CheckpointManager:
    def __init__(self, cfg):
        self.dir = cfg.checkpoint.dir
        self.steps = set(cfg.checkpoint.save_steps)
        os.makedirs(self.dir, exist_ok=True)

    def path(self, step: int) -> str:
        return os.path.join(self.dir, f"ckpt_step_{step:06d}.pt")

    def maybe_save(self, step, state, logger=None):
        if step in self.steps:
            dst = self.path(step)
            torch.save(state, dst)
            print(f"[checkpoint] saved -> {dst}")
            if logger and hasattr(logger, "add_artifact"):
                logger.add_artifact(dst, f"ckpt-{step}")


class Trainer:
    def __init__(self, cfg, model, mlm_head, dataset, masking, logger=None):
        self.cfg = cfg
        self.model = model
        self.mlm_head = mlm_head
        self.dataset = dataset
        self.masking = masking
        self.logger = logger
        self.seq_len = dataset.seq_len
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

    def _forward_features(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(images)
        else:
            feats = self.model(images)
        if isinstance(feats, tuple):
            feats = feats[0]
        if feats.ndim == 3 and feats.shape[1] == self.seq_len + 1:
            return feats[:, 1:, :]
        if feats.ndim == 3 and feats.shape[1] == self.seq_len:
            return feats
        raise RuntimeError(
            f"Unexpected feature shape {tuple(feats.shape)} for seq_len={self.seq_len}"
        )

    def step_once(self, batch_ids: torch.Tensor):
        batch_ids = batch_ids.to(self.device)
        corrupt, target, mask = self.masking(batch_ids)
        images = self.dataset.tokens_to_image_batch(corrupt).to(self.device)
        feats = self._forward_features(images)
        logits = self.mlm_head(feats)
        masked_logits = logits[mask]
        masked_targets = target[mask]
        if masked_targets.numel() == 0:
            return None
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
        iterator = iter(self.loader)
        for step in range(1, self.cfg.training.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.loader)
                batch = next(iterator)
            result = self.step_once(batch)
            if result is None:
                continue
            loss, acc, lr = result
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

