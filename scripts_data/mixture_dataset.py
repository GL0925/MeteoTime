from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from config_data import DataConfig

from .arrow_store import ArrowShardCache
from .collate import weather_collate


ERA5_VARIABLE_CATEGORIES: dict[str, tuple[int, ...]] = {
    "temperature_surface": (0,),
    "temperature_upper_air": tuple(range(24, 31)),
    "relative_humidity": tuple(range(10, 17)),
    "wind_speed": (1, 2, *range(31, 45)),
}

ERA5_PRESSURE_VARIABLE_CATEGORIES: dict[str, tuple[int, ...]] = {
    "pressure": (0, 1),
}

BTS_VARIABLE_CATEGORIES = {
    "temp_c": "temperature_surface",
    "relative_humidity_pct": "relative_humidity",
    "wind_speed_ms": "wind_speed",
    "sea_level_pressure_hpa": "pressure",
}


def source_variable_categories(
    source_name: str, item_ids: np.ndarray, num_variates: np.ndarray
) -> dict[str, tuple[np.ndarray, tuple[int, ...]]]:
    """Return record indices and variable indices for canonical forecast targets."""
    if source_name == "era5":
        if not np.all(num_variates == 45):
            raise ValueError("ERA5 records must expose the documented 45 variables")
        records = np.arange(len(item_ids), dtype=np.int64)
        return {
            name: (records, variable_ids)
            for name, variable_ids in ERA5_VARIABLE_CATEGORIES.items()
        }
    if source_name == "era5_pressure":
        if not np.all(num_variates == 2):
            raise ValueError("ERA5 pressure records must expose 2 variables")
        records = np.arange(len(item_ids), dtype=np.int64)
        return {
            name: (records, variable_ids)
            for name, variable_ids in ERA5_PRESSURE_VARIABLE_CATEGORIES.items()
        }
    if source_name == "bts_airport_weather":
        grouped: dict[str, list[int]] = {name: [] for name in set(BTS_VARIABLE_CATEGORIES.values())}
        for index, item_id in enumerate(item_ids):
            feature = str(item_id).rsplit(":", 1)[-1]
            category = BTS_VARIABLE_CATEGORIES.get(feature)
            if category is not None:
                grouped[category].append(index)
        return {
            name: (np.asarray(indices, dtype=np.int64), (0,))
            for name, indices in grouped.items()
            if indices
        }
    if source_name == "wind_farms_with_missing":
        return {"wind_speed": (np.arange(len(item_ids), dtype=np.int64), (0,))}
    return {}


@dataclass(frozen=True)
class SourceCatalog:
    name: str
    files: tuple[Path, ...]
    file_ids: np.ndarray
    row_indices: np.ndarray
    item_ids: np.ndarray
    lengths: np.ndarray
    num_variates: np.ndarray
    category_records: dict[str, np.ndarray]
    category_variable_ids: dict[str, tuple[int, ...]]

    @classmethod
    def load(cls, name: str, config: DataConfig) -> "SourceCatalog":
        source_meta = config.meta_path / name
        manifest = pq.read_table(source_meta / "manifest.parquet")
        eligible = manifest.column("eligible").to_numpy(zero_copy_only=False).astype(bool)
        manifest = manifest.filter(eligible)
        if manifest.num_rows == 0:
            raise ValueError(f"{name} has no eligible series")
        payload = json.loads((source_meta / "files.json").read_text(encoding="utf-8"))
        files = []
        legacy_root = Path("datasets/Train_data")
        for path_value in payload["files"]:
            path = Path(path_value)
            if path == legacy_root or legacy_root in path.parents:
                path = config.raw_path / path.relative_to(legacy_root)
            else:
                path = config.project_root / path
            files.append(path.resolve())
        files = tuple(files)
        item_ids = np.asarray(manifest.column("item_id").to_pylist(), dtype=object)
        num_variates = manifest.column("num_variates").to_numpy(zero_copy_only=False)
        categories = source_variable_categories(name, item_ids, num_variates)
        return cls(
            name=name,
            files=files,
            file_ids=manifest.column("file_id").to_numpy(zero_copy_only=False),
            row_indices=manifest.column("row_idx").to_numpy(zero_copy_only=False),
            item_ids=item_ids,
            lengths=manifest.column("length").to_numpy(zero_copy_only=False),
            num_variates=num_variates,
            category_records={name: records for name, (records, _) in categories.items()},
            category_variable_ids={name: variable_ids for name, (_, variable_ids) in categories.items()},
        )


class WeatherMixtureDataset(IterableDataset):
    """Random, source-balanced windows for pretraining and forecasting."""

    def __init__(
        self,
        config: DataConfig | None = None,
        samples_per_epoch: int = 100_000,
        seed: int = 2026,
        max_open_files: int | None = None,
        min_context_observed: float | None = None,
        min_target_observed: float | None = None,
        normalize: bool | None = None,
        rank: int | None = None,
        split: Literal["train", "validation"] = "train",
    ) -> None:
        super().__init__()
        self.config = config or DataConfig()
        self.config.validate()
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.max_open_files = (
            self.config.max_open_files if max_open_files is None else max_open_files
        )
        self.min_context_observed = (
            self.config.min_context_observed
            if min_context_observed is None
            else min_context_observed
        )
        self.min_target_observed = (
            self.config.min_target_observed
            if min_target_observed is None
            else min_target_observed
        )
        self.normalize = self.config.normalize if normalize is None else normalize
        self.rank = rank
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        self.split = split

        catalogs: list[SourceCatalog] = []
        candidate_names = list(self.config.source_weights)
        candidate_names.extend(
            name
            for name, group in self.config.source_aliases.items()
            if group in self.config.source_weights and name not in candidate_names
        )
        for name in candidate_names:
            group_name = self.config.source_aliases.get(name, name)
            weight = self.config.source_weights.get(name, self.config.source_weights.get(group_name, 0.0))
            source_meta = self.config.meta_path / name
            if (
                self.config.source_weights.get(group_name, weight) > 0
                and (source_meta / "manifest.parquet").exists()
            ):
                catalogs.append(SourceCatalog.load(name, self.config))
        if not catalogs:
            raise FileNotFoundError(f"no prepared source manifests under {self.config.meta_path}")
        self.catalogs = tuple(catalogs)
        self.catalog_groups = tuple(
            self.config.source_aliases.get(catalog.name, catalog.name)
            for catalog in self.catalogs
        )
        self.source_group_names = tuple(self.config.source_weights)
        self.source_group_probabilities = np.asarray(
            [self.config.source_weights[name] for name in self.source_group_names],
            dtype=np.float64,
        )
        self.source_group_probabilities /= self.source_group_probabilities.sum()
        available_lengths = tuple(
            (
                np.maximum(catalog.lengths - self.config.validation_length, 0)
                if self.split == "train"
                else catalog.lengths
            )
            for catalog in self.catalogs
        )
        self.eligible_records = {
            context_length: tuple(
                np.flatnonzero(
                    lengths
                    >= context_length + self.config.prediction_length
                )
                for lengths in available_lengths
            )
            for context_length in self.config.context_lengths
        }
        self.category_eligible_records = {
            context_length: {
                category: tuple(
                    np.intersect1d(
                        self.eligible_records[context_length][source_id],
                        catalog.category_records.get(category, np.empty(0, dtype=np.int64)),
                        assume_unique=True,
                    )
                    for source_id, catalog in enumerate(self.catalogs)
                )
                for category in self.config.target_variable_weights
            }
            for context_length in self.config.context_lengths
        }
        self.target_variable_names = tuple(self.config.target_variable_weights)
        self.target_variable_probabilities = np.asarray(
            [self.config.target_variable_weights[name] for name in self.target_variable_names],
            dtype=np.float64,
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(int(epoch))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _distributed_rank(self) -> int:
        if self.rank is not None:
            return self.rank
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def _sampling_epoch(self) -> int:
        return int(self._epoch.item()) if self.split == "train" else 0

    def _choose_context_length(self, series_length: int, rng: np.random.Generator) -> int:
        lengths = np.asarray(self.config.context_lengths, dtype=np.int64)
        probabilities = np.asarray(self.config.context_probabilities, dtype=np.float64)
        allowed = lengths + self.config.prediction_length <= series_length
        if not allowed.any():
            raise ValueError("series is shorter than the minimum training window")
        probabilities = np.where(allowed, probabilities, 0.0)
        probabilities /= probabilities.sum()
        return int(rng.choice(lengths, p=probabilities))

    def _choose_target_variable(
        self, context_length: int, rng: np.random.Generator
    ) -> str | None:
        available = np.asarray(
            [
                any(len(records) > 0 for records in self.category_eligible_records[context_length][name])
                for name in self.target_variable_names
            ],
            dtype=bool,
        )
        if not available.any():
            return None
        probabilities = np.where(available, self.target_variable_probabilities, 0.0)
        probabilities /= probabilities.sum()
        return str(rng.choice(self.target_variable_names, p=probabilities))

    def _choose_catalog_for_target(
        self, category: str, context_length: int, rng: np.random.Generator
    ) -> tuple[int, np.ndarray, tuple[int, ...]]:
        records_by_source = self.category_eligible_records[context_length][category]
        available_catalogs = np.asarray(
            [len(records) > 0 for records in records_by_source], dtype=bool
        )
        available_groups = np.asarray(
            [
                any(
                    available_catalogs[catalog_id]
                    for catalog_id, group in enumerate(self.catalog_groups)
                    if group == group_name
                )
                for group_name in self.source_group_names
            ],
            dtype=bool,
        )
        probabilities = np.where(available_groups, self.source_group_probabilities, 0.0)
        if not probabilities.any():
            raise RuntimeError(f"no source available for target category {category}")
        probabilities /= probabilities.sum()
        group_name = str(rng.choice(self.source_group_names, p=probabilities))
        candidates = np.flatnonzero(
            available_catalogs
            & np.asarray([group == group_name for group in self.catalog_groups], dtype=bool)
        )
        source_id = int(rng.choice(candidates))
        return (
            source_id,
            records_by_source[source_id],
            self.catalogs[source_id].category_variable_ids[category],
        )

    def _sample(
        self,
        rng: np.random.Generator,
        cache: ArrowShardCache,
        context_length: int | None = None,
    ) -> dict[str, torch.Tensor | int | float | str]:
        for _ in range(64):
            if context_length is None:
                minimum_context = min(self.config.context_lengths)
            else:
                minimum_context = context_length
            category = self._choose_target_variable(minimum_context, rng)
            if category is None:
                available = np.asarray(
                    [len(records) > 0 for records in self.eligible_records[minimum_context]],
                    dtype=bool,
                )
                probabilities = np.where(
                    [
                        any(
                            available[catalog_id]
                            for catalog_id, group in enumerate(self.catalog_groups)
                            if group == group_name
                        )
                        for group_name in self.source_group_names
                    ],
                    self.source_group_probabilities,
                    0.0,
                )
                probabilities /= probabilities.sum()
                group_name = str(rng.choice(self.source_group_names, p=probabilities))
                candidates = np.flatnonzero(
                    available
                    & np.asarray(
                        [group == group_name for group in self.catalog_groups], dtype=bool
                    )
                )
                source_id = int(rng.choice(candidates))
                eligible_records = self.eligible_records[minimum_context][source_id]
                if len(eligible_records) == 0:
                    continue
                variable_ids = None
            else:
                source_id, eligible_records, variable_ids = self._choose_catalog_for_target(
                    category, minimum_context, rng
                )
            catalog = self.catalogs[source_id]
            record_idx = int(rng.choice(eligible_records))
            length = int(catalog.lengths[record_idx])
            split_point = max(0, length - self.config.validation_length)
            available_length = split_point if self.split == "train" else length
            selected_context_length = (
                self._choose_context_length(available_length, rng)
                if context_length is None
                else context_length
            )
            if variable_ids is None:
                variable_id = int(rng.integers(int(catalog.num_variates[record_idx])))
            else:
                variable_id = int(rng.choice(variable_ids))
            file_path = catalog.files[int(catalog.file_ids[record_idx])]
            values = cache.read_series(
                file_path, int(catalog.row_indices[record_idx]), variable_id
            )
            window_length = selected_context_length + self.config.prediction_length
            if self.split == "train":
                minimum_start = 0
                maximum_start = split_point - window_length
            else:
                minimum_start = max(0, split_point - selected_context_length)
                maximum_start = length - window_length
            if maximum_start < minimum_start:
                continue
            window_start = int(rng.integers(minimum_start, maximum_start + 1))
            window = values[window_start : window_start + window_length]
            context = window[:selected_context_length]
            target = window[selected_context_length:]
            context_mask = np.isfinite(context)
            target_mask = np.isfinite(target)
            if (
                self.split == "train"
                and rng.random() < self.config.prefix_mask_probability
            ):
                max_prefix = int(
                    selected_context_length * self.config.prefix_mask_max_ratio
                )
                prefix_length = int(rng.integers(max_prefix + 1))
                context_mask[:prefix_length] = False
            if context_mask.mean() < self.min_context_observed:
                continue
            if target_mask.mean() < self.min_target_observed:
                continue

            observed_context = context[context_mask].astype(np.float64)
            location = float(observed_context.mean())
            # TimesFM 使用无偏标准差；仅对观测值统计以避免缺失填充值污染。
            scale = float(observed_context.std(ddof=1))
            if not np.isfinite(scale):
                continue
            if scale < self.config.normalization_scale_floor:
                scale = 1.0
            if self.normalize:
                context_values = (context - location) / scale
                target_values = (target - location) / scale
            else:
                context_values = context.copy()
                target_values = target.copy()
            context_values[~context_mask] = 0.0
            target_values[~target_mask] = 0.0
            return {
                "context": torch.from_numpy(context_values.astype(np.float32)),
                "target": torch.from_numpy(target_values.astype(np.float32)),
                "context_mask": torch.from_numpy(context_mask),
                "target_mask": torch.from_numpy(target_mask),
                "loc": torch.tensor(location, dtype=torch.float32),
                "scale": torch.tensor(scale, dtype=torch.float32),
                "source_id": source_id,
                "source_name": catalog.name,
                "target_variable": category or "unclassified",
                "variable_id": variable_id,
                "item_id": str(catalog.item_ids[record_idx]),
                "window_start": window_start,
                "context_length": selected_context_length,
                "split_point": split_point,
                "split": self.split,
            }
        raise RuntimeError("failed to find a valid window after 64 attempts")

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rank = self._distributed_rank()
        seed = self.seed + self._sampling_epoch() * 1_000_003 + rank * 10_007 + worker_id
        rng = np.random.default_rng(seed)
        cache = ArrowShardCache(self.max_open_files)
        try:
            for _ in range(worker_id, self.samples_per_epoch, num_workers):
                yield self._sample(rng, cache)
        finally:
            cache.close()


class WeatherBatchDataset(WeatherMixtureDataset):
    """Yield complete, equal-length batches so causal SDPA needs no padding mask."""

    def __init__(
        self,
        config: DataConfig | None = None,
        batches_per_epoch: int = 1_000,
        batch_size: int = 128,
        context_schedule: tuple[int, ...] | None = None,
        **kwargs,
    ) -> None:
        if batches_per_epoch <= 0 or batch_size <= 0:
            raise ValueError("batches_per_epoch and batch_size must be positive")
        super().__init__(
            config=config,
            samples_per_epoch=batches_per_epoch * batch_size,
            **kwargs,
        )
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        if context_schedule is not None:
            if not context_schedule:
                raise ValueError("context_schedule cannot be empty")
            if any(length not in self.config.context_lengths for length in context_schedule):
                raise ValueError("context_schedule contains an unconfigured length")
        self.context_schedule = context_schedule

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _batch_context_length(self, batch_idx: int) -> int:
        if self.context_schedule is not None:
            return self.context_schedule[batch_idx % len(self.context_schedule)]
        epoch = self._sampling_epoch()
        rng = np.random.default_rng(self.seed + epoch * 1_000_003 + batch_idx)
        return int(
            rng.choice(
                self.config.context_lengths,
                p=self.config.context_probabilities,
            )
        )

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rank = self._distributed_rank()
        epoch = self._sampling_epoch()
        cache = ArrowShardCache(self.max_open_files)
        try:
            for batch_idx in range(worker_id, self.batches_per_epoch, num_workers):
                context_length = self._batch_context_length(batch_idx)
                sample_seed = (
                    self.seed
                    + epoch * 1_000_003
                    + rank * 10_007
                    + batch_idx * 97
                )
                rng = np.random.default_rng(sample_seed)
                samples = [
                    self._sample(rng, cache, context_length=context_length)
                    for _ in range(self.batch_size)
                ]
                yield weather_collate(samples)
        finally:
            cache.close()
