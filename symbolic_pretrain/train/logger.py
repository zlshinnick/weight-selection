"""Logging utilities for training metrics and experiment tracking."""


class StdoutLogger:
    """Simple logger that prints training metrics to standard output."""

    def log(self, step, loss, acc, lr, extra=None):
        """Log training metrics to stdout.

        Args:
            step: Current training step number.
            loss: Masked language modeling loss value.
            acc: Mask prediction accuracy.
            lr: Current learning rate.
            extra: Optional dictionary of additional metrics (not used for stdout).
        """
        print(
            f"step {step:05d} | mlm_loss {loss:.4f} | mask_acc {acc:.3f} | lr {lr:.2e}"
        )


class WandbLogger:
    """Logger that tracks experiments using Weights & Biases.

    This logger initializes a W&B run and provides methods for logging metrics,
    watching model gradients, and saving artifacts.
    """

    def __init__(self, cfg):
        """Initialize the W&B logger and create a new run.

        Args:
            cfg: Configuration object containing:
                - wandb.project: W&B project name
                - wandb.name: Run name
                - wandb.tags: List of tags for the run
                - wandb.notes: Description/notes for the run
                - wandb.entity: Optional W&B entity (team or username)
        """
        import wandb

        init_kwargs = {
            "project": cfg.wandb.project,
            "config": cfg,
            "name": cfg.wandb.name,
            "tags": cfg.wandb.tags,
            "notes": cfg.wandb.notes,
        }
        if cfg.wandb.entity:
            init_kwargs["entity"] = cfg.wandb.entity
        wandb.init(**init_kwargs)
        self.wandb = wandb

    def watch(self, model, freq):
        """Enable W&B model watching to track gradients and parameters.

        Args:
            model: PyTorch model to watch.
            freq: Logging frequency for gradients and parameters.
        """
        self.wandb.watch(model, log="all", log_freq=freq)

    def log(self, step, loss, acc, lr, extra=None):
        """Log training metrics to W&B.

        Args:
            step: Current training step number.
            loss: Masked language modeling loss value.
            acc: Mask prediction accuracy.
            lr: Current learning rate.
            extra: Optional dictionary of additional metrics to log.
        """
        payload = {
            "train/loss": loss,
            "train/accuracy": acc,
            "train/step": step,
            "train/learning_rate": lr,
        }
        if extra:
            payload.update(extra)
        self.wandb.log(payload, step=step)

    def add_artifact(self, path, name):
        """Add a file artifact (e.g., model checkpoint) to W&B.

        Args:
            path: Path to the file to upload.
            name: Name for the artifact in W&B.
        """
        art = self.wandb.Artifact(name=name, type="model")
        art.add_file(path)
        self.wandb.log_artifact(art)

    def finish(self):
        """Finish the W&B run and upload any remaining data."""
        self.wandb.finish()
