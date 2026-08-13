from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from config_eval import EvalConfig
from config_model import ModelConfig
from models.meteotime import MeteoTime


WEATHER_FEATURES = {
    "wind_speed": ("wv (m/s)", "m/s"),
    "pressure": ("p (mbar)", "mbar"),
    "temperature": ("T (degC)", "degC"),
    "relative_humidity": ("rh (%)", "%"),
}
MARINE_STATIONS = ("DSN", "XMD", "XCS", "SSN")
MARINE_FEATURES = {
    "wind_speed": (slice(60, 64), "m/s"),
    "pressure": (slice(64, 71), "mbar"),
    "temperature": (slice(45, 50), "degC"),
}
SEGMENTS = (("0-6h", 0, 6), ("6-12h", 6, 12), ("12-18h", 12, 18), ("18-24h", 18, 24), ("24-48h", 24, 48))
EVAL_CONTEXT_LENGTHS = (128, 256, 512, 1024)


@dataclass(frozen=True)
class SeriesData:
    dataset: str
    series_id: str
    feature: str
    unit: str
    values: np.ndarray
    test_start: int


@dataclass(frozen=True)
class EvaluationWindows:
    contexts: np.ndarray
    targets: np.ndarray
    dataset: np.ndarray
    series_id: np.ndarray
    feature: np.ndarray
    unit: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed MeteoTime evaluation protocol")
    parser.add_argument("--device", default=EvalConfig.device)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("meteotime", "timesfm_2_5", "chronos2"),
        default=("meteotime", "timesfm_2_5", "chronos2"),
        help="models to run; persistence and seasonal-naive baselines always run",
    )
    parser.add_argument("--output", type=Path, default=EvalConfig.output_path)
    return parser.parse_args()


def read_scaler(path: Path) -> dict[str, tuple[float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(zip(payload["feature_names"], zip(payload["mean"], payload["std"])))


def load_weather(config: EvalConfig) -> list[SeriesData]:
    # 公开 weather 基准只使用 test split；其前 168 小时仅作为首次预测的历史输入。
    frame = pd.read_parquet(config.weather_dir / "test.parquet")
    scaler = read_scaler(config.weather_dir / "scaler_params.json")
    result: list[SeriesData] = []
    for feature, (column, unit) in WEATHER_FEATURES.items():
        mean, std = scaler[column]
        raw = frame[column].to_numpy(dtype=np.float64) * std + mean
        hourly = pd.Series(raw).groupby(np.arange(len(raw)) // 6).mean().to_numpy(dtype=np.float32)
        result.append(SeriesData("weather", "jena_2020", feature, unit, hourly, 0))
    return result


def parse_number(text: str) -> float:
    try:
        value = float(text.strip())
    except ValueError:
        return np.nan
    return np.nan if abs(value) >= 9999.0 else value


def load_marine_station(root: Path, station: str, config: EvalConfig) -> list[SeriesData]:
    rows: list[tuple[pd.Timestamp, float, float, float]] = []
    for path in sorted((root / station).glob(f"{station}*.txt")):
        suffix = path.stem[len(station) :]
        year, month = 2000 + int(suffix[:2]), int(suffix[2:])
        for line in path.read_text(encoding="utf-8").splitlines():
            if len(line) < 71:
                continue
            try:
                timestamp = pd.Timestamp(year, month, int(line[17:19]), int(line[21:23]))
            except (TypeError, ValueError):
                continue
            rows.append(
                (
                    timestamp,
                    parse_number(line[60:64]),
                    parse_number(line[64:71]),
                    parse_number(line[45:50]),
                )
            )
    if not rows:
        raise ValueError(f"no parseable rows for marine station {station}")
    frame = pd.DataFrame(rows, columns=("timestamp", "wind_speed", "pressure", "temperature"))
    frame = frame.groupby("timestamp", as_index=True).mean().sort_index()
    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    frame = frame.reindex(full_index)
    test_start = len(frame) - int(len(frame) * config.test_fraction)
    return [
        SeriesData("marine_stations", station, feature, unit, frame[feature].to_numpy(np.float32), test_start)
        for feature, (_, unit) in MARINE_FEATURES.items()
    ]


def load_marine(config: EvalConfig) -> list[SeriesData]:
    result: list[SeriesData] = []
    for station in MARINE_STATIONS:
        result.extend(load_marine_station(config.marine_root, station, config))
    return result


def load_wind(config: EvalConfig) -> list[SeriesData]:
    partials: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        config.wind_benchmark_path,
        usecols=("TurbID", "Day", "Tmstamp", "Wspd"),
        chunksize=250_000,
    ):
        clock = chunk["Tmstamp"].astype(str).str.extract(r"^(\d{1,2}):(\d{2})$")
        day = pd.to_numeric(chunk["Day"], errors="coerce")
        hour = pd.to_numeric(clock[0], errors="coerce")
        minute = pd.to_numeric(clock[1], errors="coerce")
        speed = pd.to_numeric(chunk["Wspd"], errors="coerce").where(lambda value: value >= 0)
        valid = day.ge(1) & hour.between(0, 23) & minute.between(0, 59)
        partials.append(
            pd.DataFrame(
                {
                    "station_id": chunk["TurbID"],
                    "hour": ((day - 1) * 24 + hour).where(valid),
                    "speed": speed,
                }
            )
            .dropna()
            .groupby(["station_id", "hour"], as_index=False)["speed"]
            .agg(("sum", "count"))
            .reset_index()
        )
    grouped = pd.concat(partials, ignore_index=True).groupby(["station_id", "hour"], as_index=False)[["sum", "count"]].sum()
    result: list[SeriesData] = []
    for station_id, frame in grouped.groupby("station_id", sort=True):
        length = int(frame["hour"].max()) + 1
        values = np.full(length, np.nan, dtype=np.float32)
        valid = frame["count"] >= config.min_wind_points_per_hour
        values[frame.loc[valid, "hour"].astype(np.int64)] = (
            frame.loc[valid, "sum"] / frame.loc[valid, "count"]
        ).to_numpy(dtype=np.float32)
        result.append(
            SeriesData(
                "wind_farms",
                f"turbine_{int(station_id)}",
                "wind_speed",
                "m/s",
                values,
                len(values) - int(len(values) * config.test_fraction),
            )
        )
    return result


def build_window_origins(
    series: list[SeriesData], config: EvalConfig, max_context_length: int
) -> list[tuple[int, int]]:
    """固定预测起点；所有上下文长度复用同一批 origin。"""
    origins: list[tuple[int, int]] = []
    for series_index, item in enumerate(series):
        start = max(item.test_start, max_context_length)
        for origin in range(start, len(item.values) - config.prediction_length + 1, config.stride):
            context = item.values[origin - max_context_length : origin]
            target = item.values[origin : origin + config.prediction_length]
            if np.isfinite(context).mean() < config.min_context_observed:
                continue
            if np.isfinite(target).mean() < config.min_target_observed:
                continue
            origins.append((series_index, origin))
    if not origins:
        raise ValueError("no valid evaluation windows")
    return origins


def build_windows(
    series: list[SeriesData], config: EvalConfig, origins: list[tuple[int, int]]
) -> EvaluationWindows:
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[tuple[str, str, str, str]] = []
    for series_index, origin in origins:
        item = series[series_index]
        context = item.values[origin - config.context_length : origin]
        target = item.values[origin : origin + config.prediction_length]
        contexts.append(context)
        targets.append(target)
        metadata.append((item.dataset, item.series_id, item.feature, item.unit))
    if not contexts:
        raise ValueError("no valid evaluation windows")
    columns = tuple(np.asarray(values, dtype=object) for values in zip(*metadata))
    return EvaluationWindows(np.stack(contexts), np.stack(targets), *columns)


def impute_contexts(contexts: np.ndarray) -> np.ndarray:
    result = np.empty_like(contexts)
    for index, values in enumerate(contexts):
        result[index] = pd.Series(values).interpolate(limit_direction="both").fillna(0.0).to_numpy(np.float32)
    return result


def persistence(contexts: np.ndarray, horizon: int) -> np.ndarray:
    values = impute_contexts(contexts)
    return np.repeat(values[:, -1:], horizon, axis=1)


def seasonal_naive(contexts: np.ndarray, horizon: int) -> np.ndarray:
    values = impute_contexts(contexts)
    return np.tile(values[:, -24:], (1, (horizon + 23) // 24))[:, :horizon]


def metric_block(
    predictions: np.ndarray,
    targets: np.ndarray,
    contexts: np.ndarray,
    intervals: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    valid = np.isfinite(predictions) & np.isfinite(targets)
    errors = predictions[valid] - targets[valid]
    output: dict[str, float | int | None] = {
        "mae": float(np.mean(np.abs(errors), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(errors), dtype=np.float64))),
        "observed_points": int(errors.size),
    }
    scaled_errors: list[float] = []
    for forecast, target, context in zip(predictions, targets, contexts):
        target_mask = np.isfinite(forecast) & np.isfinite(target)
        seasonal_mask = np.isfinite(context[24:]) & np.isfinite(context[:-24])
        if target_mask.any() and seasonal_mask.any():
            denominator = np.mean(np.abs(context[24:][seasonal_mask] - context[:-24][seasonal_mask]))
            if denominator > 1e-8:
                scaled_errors.append(float(np.mean(np.abs(forecast[target_mask] - target[target_mask])) / denominator))
    output["mase"] = float(np.mean(scaled_errors)) if scaled_errors else None
    if intervals is None:
        output["p05_p95_coverage"] = None
    else:
        interval_mask = np.isfinite(targets) & np.isfinite(intervals[..., 0]) & np.isfinite(intervals[..., 1])
        covered = (targets >= intervals[..., 0]) & (targets <= intervals[..., 1])
        output["p05_p95_coverage"] = float(np.mean(covered[interval_mask])) if interval_mask.any() else None
    return output


def summarize(
    predictions: np.ndarray,
    windows: EvaluationWindows,
    intervals: np.ndarray | None = None,
) -> dict[str, dict[str, float | int | None]]:
    result = {"overall": metric_block(predictions, windows.targets, windows.contexts, intervals)}
    for name, start, end in SEGMENTS:
        subset_intervals = intervals[:, start:end] if intervals is not None else None
        result[name] = metric_block(
            predictions[:, start:end], windows.targets[:, start:end], windows.contexts, subset_intervals
        )
    return result


def group_results(
    predictions: np.ndarray,
    windows: EvaluationWindows,
    intervals: np.ndarray | None = None,
) -> dict[str, dict[str, dict[str, dict[str, float | int | None]]]]:
    result: dict[str, dict[str, dict[str, dict[str, float | int | None]]]] = {}
    for dataset in sorted(set(windows.dataset)):
        for feature in sorted(set(windows.feature[windows.dataset == dataset])):
            index = (windows.dataset == dataset) & (windows.feature == feature)
            subset = EvaluationWindows(
                windows.contexts[index], windows.targets[index], windows.dataset[index], windows.series_id[index], windows.feature[index], windows.unit[index]
            )
            subset_intervals = intervals[index] if intervals is not None else None
            result.setdefault(str(dataset), {})[str(feature)] = summarize(predictions[index], subset, subset_intervals)
    return result


def predict_meteotime(
    windows: EvaluationWindows, config: EvalConfig
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...], np.ndarray]:
    checkpoint = torch.load(config.meteotime_checkpoint, map_location="cpu", weights_only=False)
    model = MeteoTime(ModelConfig(**checkpoint["model_config"])).to(config.device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(windows.contexts), config.meteotime_batch_size),
            desc="MeteoTime",
            unit="batch",
        ):
            values = torch.from_numpy(windows.contexts[start : start + config.meteotime_batch_size]).to(config.device)
            mask = torch.isfinite(values)
            with torch.autocast(device_type=torch.device(config.device).type, dtype=torch.bfloat16, enabled="cuda" in config.device):
                outputs.append(model.forecast(values, config.prediction_length, mask)["quantiles"].float().cpu().numpy())
    quantiles = np.concatenate(outputs, axis=0)
    indices = [model.config.quantiles.index(level) for level in config.quantile_levels]
    selected = quantiles[..., indices]
    median = selected[..., config.quantile_levels.index(0.5)]
    intervals = quantiles[..., [model.config.quantiles.index(0.05), model.config.quantiles.index(0.95)]]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return median, selected, config.quantile_levels, intervals


def predict_timesfm(
    windows: EvaluationWindows, config: EvalConfig
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...], None]:
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(config.timesfm_path, local_files_only=True, torch_compile=False)
    model.compile(
        timesfm.ForecastConfig(
            max_context=config.context_length,
            max_horizon=config.prediction_length,
            normalize_inputs=True,
            per_core_batch_size=config.timesfm_batch_size,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
        )
    )
    contexts = impute_contexts(windows.contexts)
    outputs: list[np.ndarray] = []
    for start in tqdm(
        range(0, len(contexts), model.global_batch_size),
        desc="TimesFM 2.5",
        unit="batch",
    ):
        _, quantiles = model.forecast(config.prediction_length, [row.copy() for row in contexts[start : start + model.global_batch_size]])
        outputs.append(quantiles[..., 1:])
    all_quantiles = np.concatenate(outputs, axis=0).astype(np.float32)
    native_levels = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    selected = all_quantiles[..., [native_levels.index(level) for level in config.quantile_levels]]
    median = selected[..., config.quantile_levels.index(0.5)]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return median, selected, config.quantile_levels, None


def predict_chronos2(
    windows: EvaluationWindows, config: EvalConfig
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...], np.ndarray]:
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(config.chronos2_path, device_map=config.device, local_files_only=True)
    contexts = impute_contexts(windows.contexts)
    timestamps = pd.date_range("2000-01-01", periods=config.context_length, freq="h")
    outputs: list[np.ndarray] = []
    all_levels = (0.05, *config.quantile_levels, 0.95)
    for start in tqdm(
        range(0, len(contexts), config.chronos2_batch_size),
        desc="Chronos2",
        unit="batch",
    ):
        batch = contexts[start : start + config.chronos2_batch_size]
        frame = pd.concat(
            [pd.DataFrame({"item_id": item_id, "timestamp": timestamps, "target": values}) for item_id, values in enumerate(batch)],
            ignore_index=True,
        )
        forecast = pipeline.predict_df(
            frame,
            prediction_length=config.prediction_length,
            quantile_levels=list(all_levels),
            batch_size=len(batch),
            context_length=config.context_length,
            cross_learning=False,
            freq="h",
        )
        outputs.append(np.stack([forecast[str(level)].to_numpy(np.float32).reshape(len(batch), -1) for level in all_levels], axis=-1))
    all_quantiles = np.concatenate(outputs, axis=0)
    selected = all_quantiles[..., 1:-1]
    median = selected[..., config.quantile_levels.index(0.5)]
    intervals = all_quantiles[..., [0, -1]]
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return median, selected, config.quantile_levels, intervals


def checkpoint_metadata(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path),
        "optimizer_step": int(checkpoint.get("global_step", 0)),
        "epoch": int(checkpoint.get("epoch", 0)),
        "best_validation_loss": checkpoint.get("best_validation_loss"),
        "model_config": checkpoint["model_config"],
        "parameters": int(sum(value.numel() for value in checkpoint["model"].values())),
    }


def print_summary(results: dict[str, object]) -> None:
    labels = {
        "persistence": "persistence",
        "seasonal_naive_24h": "seasonal_naive_24h",
        "meteotime": "meteotime",
        "timesfm_2_5": "timesfm_2_5",
        "chronos2": "chronos2",
    }
    for context_length, context_result in results["by_context"].items():
        metrics = context_result["metrics"]
        print(f"\n=== Context {context_length}h: overall 48h metrics ===")
        for dataset in ("weather", "marine_stations", "wind_farms"):
            for feature in sorted(metrics["meteotime"].get(dataset, {})):
                print(f"[{dataset} / {feature}]")
                for model_name, model_metrics in metrics.items():
                    overall = model_metrics[dataset][feature]["overall"]
                    print(
                        f"  {labels[model_name]:20s} MAE={overall['mae']:.4f} "
                        f"RMSE={overall['rmse']:.4f} MASE={overall['mase']:.4f}"
                    )
        print(f"\n=== Context {context_length}h: MeteoTime by segment ===")
        for dataset in ("weather", "marine_stations", "wind_farms"):
            for feature in sorted(metrics["meteotime"].get(dataset, {})):
                print(f"[{dataset} / {feature}]")
                for segment, _, _ in SEGMENTS:
                    block = metrics["meteotime"][dataset][feature][segment]
                    print(
                        f"  {segment:>6s} MAE={block['mae']:.4f} "
                        f"RMSE={block['rmse']:.4f} MASE={block['mase']:.4f}"
                    )


def write_comparison_markdown(results: dict[str, object], output_dir: Path) -> None:
    labels = {
        "weather": "weather",
        "marine_stations": "海洋气象观测站",
        "wind_farms": "真实风电场",
        "wind_speed": "风速",
        "pressure": "气压",
        "temperature": "气温",
        "relative_humidity": "相对湿度",
        "persistence": "持续性",
        "seasonal_naive_24h": "季节性朴素",
        "meteotime": "MeteoTime",
        "timesfm_2_5": "TimesFM 2.5",
        "chronos2": "Chronos 2",
    }
    models = ("persistence", "seasonal_naive_24h", "meteotime", "timesfm_2_5", "chronos2")
    lines = ["# 模型对比评测", "", "固定预测起点，预测未来 48 小时；单元格为 MAE / RMSE / MASE。"]
    for context_length, context_result in results["by_context"].items():
        metrics = context_result["metrics"]
        lines.extend(("", f"## 上下文 {context_length} 小时", "", "| 数据集 | 特征 | " + " | ".join(labels[name] for name in models) + " |", "| --- | --- | " + " | ".join(["---"] * len(models)) + " |"))
        for dataset in ("weather", "marine_stations", "wind_farms"):
            for feature in sorted(metrics["meteotime"].get(dataset, {})):
                cells = []
                for model in models:
                    overall = metrics[model][dataset][feature]["overall"]
                    cells.append(f"{overall['mae']:.3f} / {overall['rmse']:.3f} / {overall['mase']:.3f}")
                lines.append(f"| {labels[dataset]} | {labels[feature]} | " + " | ".join(cells) + " |")
    lines.extend(("", "原始完整结果：`meteotime-v1_metrics.json`。"))
    (output_dir / "模型对比评测.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = EvalConfig(output_path=args.output, device=args.device)
    config.validate()
    for path in (config.meteotime_checkpoint, config.timesfm_path, config.chronos2_path, config.wind_benchmark_path):
        if not path.exists():
            raise FileNotFoundError(path)
    series = load_weather(config) + load_marine(config) + load_wind(config)
    context_lengths = EVAL_CONTEXT_LENGTHS
    origins = build_window_origins(series, config, max(context_lengths))
    results: dict[str, object] = {
        "protocol": {
            "context_hours": list(context_lengths),
            "prediction_hours": config.prediction_length,
            "stride_hours": config.stride,
            "mase_baseline": "24-hour seasonal naive",
        },
        "checkpoint": checkpoint_metadata(config.meteotime_checkpoint),
        "fixed_origin_count": int(len(origins)),
        "by_context": {},
    }
    for context_length in context_lengths:
        context_config = replace(config, context_length=context_length)
        windows = build_windows(series, context_config, origins)
        metrics: dict[str, object] = {}
        baseline_predictions = {
            "persistence": persistence(windows.contexts, context_config.prediction_length),
            "seasonal_naive_24h": seasonal_naive(windows.contexts, context_config.prediction_length),
        }
        for name, prediction in baseline_predictions.items():
            metrics[name] = group_results(prediction, windows)
        predictors = {
            "meteotime": predict_meteotime,
            "timesfm_2_5": predict_timesfm,
            "chronos2": predict_chronos2,
        }
        for name in args.models:
            prediction, _, _, intervals = predictors[name](windows, context_config)
            metrics[name] = group_results(prediction, windows, intervals)
        results["by_context"][str(context_length)] = {
            "window_count": int(len(windows.contexts)),
            "window_count_by_dataset": {name: int((windows.dataset == name).sum()) for name in sorted(set(windows.dataset))},
            "metrics": metrics,
        }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    import sys
    from io import StringIO
    captured = StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    print(json.dumps({"output": str(config.output_path), "fixed_origins": results["fixed_origin_count"], "contexts": context_lengths, "models": ["persistence", "seasonal_naive_24h", *args.models]}, ensure_ascii=False))
    print_summary(results)
    sys.stdout = old_stdout
    output_text = captured.getvalue()
    config.output_path.write_text(output_text, encoding="utf-8")
    print(output_text, end="")


if __name__ == "__main__":
    main()
