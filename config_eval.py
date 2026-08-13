from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalConfig:
    """统一气象评估配置。所有长度单位均为小时。"""

    weather_dir: Path = Path("/home/amax/SSD2/GL/MeteoTime_benchmark/weather")
    marine_root: Path = Path("/home/amax/SSD2/GL/MeteoTime_benchmark")
    wind_benchmark_path: Path = Path("/home/amax/SSD2/GL/MeteoTime_benchmark/wtbdata_245days.csv")
    meteotime_checkpoint: Path = Path("checkpoints/meteotime-v1/best.pt")
    timesfm_path: Path = Path("models/timesfm2.5")
    chronos2_path: Path = Path("models/chronos2")
    output_path: Path = Path("results.txt")

    context_length: int = 512
    prediction_length: int = 48
    stride: int = 48
    test_fraction: float = 0.20
    min_context_observed: float = 0.80
    min_target_observed: float = 0.80
    min_wind_points_per_hour: int = 3
    meteotime_batch_size: int = 256
    timesfm_batch_size: int = 4
    chronos2_batch_size: int = 32
    device: str = "cuda:0"

    # 三个概率模型共同使用的分位数，保证 WQL 可横向比较。
    quantile_levels: tuple[float, ...] = (
        0.10,
        0.20,
        0.30,
        0.50,
        0.70,
        0.80,
        0.90,
    )

    def validate(self) -> None:
        if self.context_length < 24 or self.prediction_length != 48:
            raise ValueError("evaluation requires at least 24h context and a 48h horizon")
        if self.stride <= 0 or self.meteotime_batch_size <= 0:
            raise ValueError("stride and batch sizes must be positive")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be in (0, 1)")
        if not 0 <= self.min_context_observed <= 1:
            raise ValueError("min_context_observed must be in [0, 1]")
        if not 0 <= self.min_target_observed <= 1:
            raise ValueError("min_target_observed must be in [0, 1]")
