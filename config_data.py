from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    project_root: Path = field(default_factory=Path.cwd)
    raw_root: Path = Path("/home/amax/SSD2/GL/MeteoTime_train_data")
    artifact_root: Path = Path("/home/amax/SSD2/GL/MeteoTime_data_artifacts")
    max_context_length: int = 1024
    # 训练标签是一个未来 64 点 horizon；推理可继续自回归生成更长 horizon。
    prediction_length: int = 64
    validation_length: int = 512
    patch_size: int = 32
    context_lengths: tuple[int, ...] = (128, 256, 512, 1024)
    context_probabilities: tuple[float, ...] = (0.15, 0.25, 0.40, 0.20)
    max_open_files: int = 12
    min_context_observed: float = 0.50
    min_target_observed: float = 0.50
    normalize: bool = True
    # 与 TimesFM 的 RevIN 一致：仅作为近常量序列的判定阈值。
    normalization_scale_floor: float = 1e-6
    prefix_mask_probability: float = 0.20
    prefix_mask_max_ratio: float = 0.30
    # 统计来源组权重；物理来源通过 source_aliases 归并到同一组。
    source_weights: dict[str, float] = field(
        default_factory=lambda: {
            "bts_airport_weather": 0.30,
            "era5": 0.70,
            # 独立存储的 ERA5 气压 Arrow 归入 era5 统计组。
            "era5_pressure": 0.0,
        }
    )
    source_aliases: dict[str, str] = field(
        default_factory=lambda: {"era5_pressure": "era5"}
    )
    # 先选择预测变量类别，再在该类别内按 source_weights 选数据源。
    # 未能识别变量语义的数据源不会参与这四类目标的预训练。
    target_variable_weights: dict[str, float] = field(
        default_factory=lambda: {
            "wind_speed": 0.20,
            "pressure": 0.20,
            "relative_humidity": 0.20,
            "temperature_surface": 0.20,
            "temperature_upper_air": 0.20,
        }
    )

    @property
    def raw_path(self) -> Path:
        return (self.project_root / self.raw_root).resolve()

    @property
    def meta_path(self) -> Path:
        return (self.project_root / self.artifact_root / "meta").resolve()

    @property
    def processed_path(self) -> Path:
        return (self.project_root / self.artifact_root / "processed").resolve()

    def validate(self) -> None:
        if self.max_context_length % self.patch_size:
            raise ValueError("max_context_length must be divisible by patch_size")
        if self.prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        if self.validation_length < self.prediction_length:
            raise ValueError("validation_length must cover at least one prediction target")
        if len(self.context_lengths) != len(self.context_probabilities):
            raise ValueError("context lengths and probabilities must have equal size")
        if max(self.context_lengths) > self.max_context_length:
            raise ValueError("context length exceeds configured maximum")
        if any(length % self.patch_size for length in self.context_lengths):
            raise ValueError("training context lengths must be divisible by patch_size")
        if abs(sum(self.context_probabilities) - 1.0) > 1e-6:
            raise ValueError("context probabilities must sum to one")
        if any(weight < 0 for weight in self.source_weights.values()):
            raise ValueError("source weights cannot be negative")
        if sum(self.source_weights.values()) <= 0:
            raise ValueError("at least one source weight must be positive")
        if not self.target_variable_weights or any(
            weight < 0 for weight in self.target_variable_weights.values()
        ):
            raise ValueError("target_variable_weights must contain non-negative weights")
        if sum(self.target_variable_weights.values()) <= 0:
            raise ValueError("at least one target variable weight must be positive")
        for source_name, group_name in self.source_aliases.items():
            if not source_name or not group_name:
                raise ValueError("source aliases must use non-empty names")
        if self.max_open_files <= 0:
            raise ValueError("max_open_files must be positive")
        if not 0 <= self.min_context_observed <= 1:
            raise ValueError("min_context_observed must be in [0, 1]")
        if not 0 <= self.min_target_observed <= 1:
            raise ValueError("min_target_observed must be in [0, 1]")
        if self.normalization_scale_floor <= 0:
            raise ValueError("normalization_scale_floor must be positive")
        if not 0 <= self.prefix_mask_probability <= 1:
            raise ValueError("prefix_mask_probability must be in [0, 1]")
        if not 0 <= self.prefix_mask_max_ratio < 1:
            raise ValueError("prefix_mask_max_ratio must be in [0, 1)")
