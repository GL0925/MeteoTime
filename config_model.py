from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    patch_size: int = 32
    # 语言模型式 next-patch 训练：每个输入 token 只预测下一个 Patch。
    forecast_patches: int = 1
    # 单个 next-token 的预测 horizon；允许不等于输入 Patch 长度。
    forecast_horizon: int = 64
    hidden_size: int = 512
    num_layers: int = 12
    num_heads: int = 8
    intermediate_size: int = 1024
    max_context_points: int = 1024
    max_sequence_tokens: int = 64
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    patch_mask_ratio: float = 0.10
    # 每隔一层启用缺失感知注意力；第一层保持标准因果注意力。
    missing_attention_every: int = 2
    missing_attention_strength: float = 0.10
    # 保留历史字段名，以兼容已保存检查点中的 model_config。
    # 语义与 TimesFM RevIN 的 tolerance 相同，不再作为标准差下限。
    normalization_scale_floor: float = 1e-6
    rope_theta: float = 10_000.0
    quantiles: tuple[float, ...] = (
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
        0.70,
        0.80,
        0.90,
        0.95,
    )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def validate(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.forecast_patches <= 0:
            raise ValueError("forecast_patches must be positive")
        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.head_dim % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.max_context_points % self.patch_size:
            raise ValueError("max_context_points must be divisible by patch_size")
        if self.max_sequence_tokens < self.max_context_points // self.patch_size:
            raise ValueError("max_sequence_tokens is smaller than the context token count")
        if not 0 <= self.patch_mask_ratio < 1:
            raise ValueError("patch_mask_ratio must be in [0, 1)")
        if self.missing_attention_every <= 0:
            raise ValueError("missing_attention_every must be positive")
        if self.missing_attention_strength < 0:
            raise ValueError("missing_attention_strength cannot be negative")
        if self.normalization_scale_floor <= 0:
            raise ValueError("normalization_scale_floor must be positive")
        if tuple(sorted(self.quantiles)) != self.quantiles:
            raise ValueError("quantiles must be sorted")
        if not all(0 < quantile < 1 for quantile in self.quantiles):
            raise ValueError("quantiles must be in (0, 1)")
        if 0.5 not in self.quantiles:
            raise ValueError("quantiles must include the median (0.5)")
