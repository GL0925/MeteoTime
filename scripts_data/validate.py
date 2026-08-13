from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from config_data import DataConfig

from .collate import weather_collate
from .mixture_dataset import WeatherMixtureDataset


def validate_manifests(config: DataConfig) -> list[dict]:
    reports = []
    for source_dir in sorted(path for path in config.meta_path.iterdir() if path.is_dir()):
        source_group = config.source_aliases.get(source_dir.name, source_dir.name)
        if config.source_weights.get(source_group, 0.0) <= 0:
            continue
        report = json.loads((source_dir / "report.json").read_text(encoding="utf-8"))
        files = json.loads((source_dir / "files.json").read_text(encoding="utf-8"))["files"]
        manifest = pq.read_table(source_dir / "manifest.parquet")
        if manifest.num_rows != report["physical_series"]:
            raise ValueError(f"{source_dir.name}: manifest row count mismatch")
        if not all((config.project_root / path).is_file() for path in files):
            raise FileNotFoundError(f"{source_dir.name}: a referenced Arrow file is missing")
        if set(manifest.column("freq").to_pylist()) != {"H"}:
            raise ValueError(f"{source_dir.name}: non-hourly record found")
        eligible = int(manifest.column("eligible").to_numpy(zero_copy_only=False).sum())
        if eligible != report["eligible_series"]:
            raise ValueError(f"{source_dir.name}: eligible count mismatch")
        reports.append(report)
    return reports


def validate_sampling(config: DataConfig, samples: int, workers: int) -> dict:
    dataset = WeatherMixtureDataset(config, samples_per_epoch=samples, seed=31)
    loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=workers,
        collate_fn=weather_collate,
    )
    source_counts: Counter[str] = Counter()
    variable_counts: Counter[str] = Counter()
    missing_points: defaultdict[str, int] = defaultdict(int)
    total_points: defaultdict[str, int] = defaultdict(int)
    observed_samples = 0
    for batch in loader:
        if batch["context"].shape[1] > config.max_context_length:
            raise ValueError("sample exceeds maximum context length")
        if batch["target"].shape[1] != config.prediction_length:
            raise ValueError("target length mismatch")
        for key in ("context", "target", "loc", "scale"):
            if not torch.isfinite(batch[key]).all():
                raise ValueError(f"non-finite value remains in {key}")
        if not (batch["scale"] > 0).all():
            raise ValueError("non-positive normalization scale")
        for index, source in enumerate(batch["source_name"]):
            source_counts[source] += 1
            variable_counts[batch["target_variable"][index]] += 1
            context_length = int(batch["context_length"][index])
            valid = int(
                batch["context_mask"][index, -context_length:].sum()
                + batch["target_mask"][index].sum()
            )
            total = context_length + batch["target_mask"][index].numel()
            missing_points[source] += total - valid
            total_points[source] += total
        observed_samples += len(batch["source_name"])
    return {
        "samples": observed_samples,
        "source_counts": dict(sorted(source_counts.items())),
        "target_variable_counts": dict(sorted(variable_counts.items())),
        "sampled_missing_ratio": {
            name: missing_points[name] / total_points[name] for name in sorted(total_points)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MeteoTime data artifacts and sampling")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = DataConfig(project_root=args.project_root.resolve())
    reports = validate_manifests(config)
    sampling = validate_sampling(config, args.samples, args.workers)
    print(json.dumps({"sources": reports, "sampling": sampling}, indent=2))


if __name__ == "__main__":
    main()
