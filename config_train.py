from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batches_per_epoch: int = 1_000
    batch_size: int = 256
    workers: int = 4
    prefetch_factor: int = 1
    pin_memory: bool = True
    learning_rate: float = 2.5e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    min_learning_rate: float = 2.5e-6
    cosine_restart_epoch: int = 30
    cosine_stage_min_ratio: float = 0.1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    seed: int = 2026
    log_every: int = 50
    validation_every_steps: int = 500
    validation_batches_per_context: int = 8
    validation_batch_size: int = 128
    validation_workers: int = 0
    validation_seed: int = 3026
    precision: str = "bf16"
    distributed_backend: str = "nccl"
    fused_optimizer: bool = True
    output_dir: Path = Path("checkpoints/meteotime-v1")
    resume_enabled: bool = False
    resume_path: Path = Path("checkpoints/meteotime-v1/best.pt")
    tensorboard_enabled: bool = True
    tensorboard_log_dir: Path = Path("runs")
    tensorboard_flush_secs: int = 10
    tensorboard_port: int = 6006

    def validate(self) -> None:
        if self.epochs <= 0 or self.batches_per_epoch <= 0:
            raise ValueError("epochs and batches_per_epoch must be positive")
        if self.batch_size <= 0 or self.workers < 0:
            raise ValueError("batch_size must be positive and workers cannot be negative")
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay cannot be negative")
        if not 0 < self.adam_beta1 < 1 or not 0 < self.adam_beta2 < 1:
            raise ValueError("Adam betas must be in (0, 1)")
        if self.gradient_accumulation <= 0 or self.max_grad_norm <= 0:
            raise ValueError("gradient_accumulation and max_grad_norm must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.validation_every_steps <= 0 or self.validation_batches_per_context <= 0:
            raise ValueError("validation interval and batches per context must be positive")
        if self.validation_batch_size <= 0 or self.validation_workers < 0:
            raise ValueError("validation batch size must be positive and workers cannot be negative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.min_learning_rate <= 0 or self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must be in (0, learning_rate]")
        if not 0 < self.cosine_stage_min_ratio < 1:
            raise ValueError("cosine_stage_min_ratio must be in (0, 1)")
        if not 0 < self.cosine_restart_epoch < self.epochs:
            raise ValueError("cosine_restart_epoch must be between 0 and epochs")
        if self.precision != "bf16":
            raise ValueError("MeteoTime V1 currently supports bf16 training only")
        if self.tensorboard_flush_secs <= 0:
            raise ValueError("tensorboard_flush_secs must be positive")
        if not 1 <= self.tensorboard_port <= 65535:
            raise ValueError("tensorboard_port must be in [1, 65535]")
