from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, StepLR


def make_scheduler(cfg, opt):
    if not cfg.scheduler.enabled:
        return None
    warmup = cfg.scheduler.warmup_steps
    total = cfg.training.steps
    if warmup > 0:
        warm = LinearLR(
            opt,
            start_factor=cfg.scheduler.warmup_start_lr / cfg.optimizer.lr,
            end_factor=1.0,
            total_iters=warmup,
        )
        if cfg.scheduler.type == "cosine":
            main = CosineAnnealingLR(
                opt, T_max=total - warmup, eta_min=cfg.scheduler.min_lr
            )
        elif cfg.scheduler.type == "step":
            main = StepLR(
                opt, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma
            )
        elif cfg.scheduler.type == "linear":
            main = LinearLR(
                opt,
                start_factor=1.0,
                end_factor=cfg.scheduler.min_lr / cfg.optimizer.lr,
                total_iters=total - warmup,
            )
        else:
            raise ValueError(f"Unknown scheduler {cfg.scheduler.type}")
        return SequentialLR(opt, schedulers=[warm, main], milestones=[warmup])
    else:
        if cfg.scheduler.type == "cosine":
            return CosineAnnealingLR(opt, T_max=total, eta_min=cfg.scheduler.min_lr)
        elif cfg.scheduler.type == "step":
            return StepLR(
                opt, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma
            )
        elif cfg.scheduler.type == "linear":
            return LinearLR(
                opt,
                start_factor=1.0,
                end_factor=cfg.scheduler.min_lr / cfg.optimizer.lr,
                total_iters=total,
            )
        else:
            raise ValueError(f"Unknown scheduler {cfg.scheduler.type}")
