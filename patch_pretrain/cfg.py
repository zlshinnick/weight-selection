from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal
import os
import yaml


@dataclass
class ModelConfig:
    name: str = "vit_tiny_patch16_224"
    num_classes: int = 0
    pretrained: bool = False
    drop_path: float = 0.0
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.0
    patch_size: int = 16
    img_size: int = 224
    in_chans: int = 1


@dataclass
class GridConfig:
    H: int = 8
    W: int = 8


@dataclass
class VocabConfig:
    K: int = 130
    PAD_ID: int = 0
    MASK_ID: int = 1


@dataclass
class DyckConfig:
    k_open: int = 64
    k_close: int = 64
    open_prob: float = 0.6
    min_pairs: int = 1


@dataclass
class DyckGaborConfig:
    n_freqs: int = 8
    n_thetas: int = 16
    freq_min: float = 0.05
    freq_max: float = 0.4
    bandwidth: float = 1.0
    sigma_x: float = 2.5
    sigma_y: float = 2.5
    n_stds: float = 3.0
    offset: float = 0.0
    component: Literal["real", "imag", "complex"] = "real"


@dataclass
class DatasetConfig:
    n_samples: int = 20000
    num_workers: int = 4
    pin_memory: bool = True
    batch_size: int = 128
    use_shuffled: bool = False


@dataclass
class TrainingConfig:
    steps: int = 50000
    mask_ratio: float = 0.5
    device: str = "cuda"


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3.5e-4
    weight_decay: float = 0.05
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class SchedulerConfig:
    enabled: bool = True
    type: str = "cosine"
    warmup_steps: int = 1000
    warmup_start_lr: float = 1.0e-7
    min_lr: float = 1.5e-5


@dataclass
class GradientClipConfig:
    enabled: bool = True
    max_norm: float = 1.0


@dataclass
class CheckpointConfig:
    dir: str = "patch_checkpoints"
    save_steps: List[int] = field(
        default_factory=lambda: [2500, 5000, 10000, 20000, 30000, 40000, 50000]
    )


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "patch_pretraining"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class LoggingConfig:
    print_freq: int = 100
    log_examples: bool = False


@dataclass
class RootConfig:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    vocab: VocabConfig = field(default_factory=VocabConfig)
    dyck: DyckConfig = field(default_factory=DyckConfig)
    dyck_gabor: DyckGaborConfig = field(default_factory=DyckGaborConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    grad_clip: GradientClipConfig = field(default_factory=GradientClipConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge(dc, dd):
    for k, v in (dd or {}).items():
        child = getattr(dc, k)
        if hasattr(child, "__dataclass_fields__"):
            _merge(child, v)
        else:
            setattr(dc, k, v)


def load_cfg(path: Optional[str]) -> RootConfig:
    cfg = RootConfig()
    if path is None:
        return cfg
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    _merge(cfg, data)
    return cfg


def cfg_to_dict(cfg: RootConfig) -> dict:
    return asdict(cfg)
