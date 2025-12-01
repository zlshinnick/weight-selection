from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
import yaml
import os


@dataclass
class ModelConfig:
    name: str = "vit_tiny_patch16_224"
    num_classes: int = 0
    pretrained: bool = False
    embed_dim: int = 192
    use_swin: bool = False
    drop_path: float = 0.0
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.0
    # Mimetic initialization flags (passed through to utils.build_model)
    mimetic_init: bool = False
    mimetic_alpha: float = 0.4
    mimetic_beta: float = 0.4
    mimetic_dist: Literal["uniform", "normal"] = "uniform"


@dataclass
class GridConfig:
    H: int = 4
    W: int = 4


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
    max_depth: int = 16


@dataclass
class WwConfig:
    alphabet: int = 64
    min_w_length: int = 3
    max_w_length: int = 32


@dataclass
class DatasetConfig:
    n_samples: int = 20000
    num_workers: int = 4
    pin_memory: bool = True
    use_shuffled: bool = False
    batch_size: int = 256
    source: Literal["dyck", "dyck_shuffle", "ww"] = "dyck"


@dataclass
class TrainingConfig:
    batch_size: int = 256
    steps: int = 50000
    mask_ratio: float = 0.5
    mask_close_only: bool = True
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
    step_size: int = 10000
    gamma: float = 0.1


@dataclass
class GradientClipConfig:
    enabled: bool = True
    max_norm: float = 1.0


@dataclass
class CheckpointConfig:
    dir: str = "checkpoints"
    save_steps: List[int] = field(
        default_factory=lambda: [2500, 5000, 10000, 20000, 30000, 40000, 50000]
    )


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "symbolic_pretraining-mlm"
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
    ww: WwConfig = field(default_factory=WwConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    grad_clip: GradientClipConfig = field(default_factory=GradientClipConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_cfg(path: Optional[str]) -> RootConfig:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "pretrain.yaml")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # shallow merge helper:
    def merge(dc, dd):
        for k, v in dd.items():
            child = getattr(dc, k)
            if hasattr(child, "__dataclass_fields__"):
                merge(child, v)
            else:
                setattr(dc, k, v)

    cfg = RootConfig()
    merge(cfg, raw or {})
    return cfg
