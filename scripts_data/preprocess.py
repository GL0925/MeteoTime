from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from config_data import DataConfig

from .resample import resample_to_hourly
from .sources import SOURCE_TYPES
from .sources.base import SourceSpec


MANIFEST_SCHEMA = pa.schema(
    [
        ("file_id", pa.int32()),
        ("row_idx", pa.int32()),
        ("item_id", pa.string()),
        ("start", pa.timestamp("s")),
        ("freq", pa.string()),
        ("length", pa.int32()),
        ("num_variates", pa.int16()),
        ("missing_ratio", pa.float32()),
        ("eligible", pa.bool_()),
    ]
)

PROCESSED_SCHEMA = pa.schema(
    [
        ("item_id", pa.string()),
        ("start", pa.timestamp("s")),
        ("freq", pa.string()),
        ("target", pa.list_(pa.float32())),
    ]
)


def _arrow_files(directories: Iterable[Path]) -> list[Path]:
    files = sorted(path for directory in directories for path in directory.glob("*.arrow"))
    if not files:
        raise FileNotFoundError("no Arrow files found")
    return files


def _read_arrow(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_stream(source).read_all()


def _relative_path(path: Path, project_root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = project_root.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        # Keep external data roots (for example /home/amax/SSD2) usable
        # when training is launched from the repository directory.
        return str(resolved_path)


def _series_shape(scalar: pa.Scalar) -> tuple[int, int]:
    values = scalar.values
    if pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
        if len(values) == 0:
            return 0, 0
        return len(values), len(values[0].values)
    return 1, len(values)


def _missing_ratio(scalar: pa.Scalar) -> float:
    values = scalar.values
    if pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
        return 0.0
    array = np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32)
    return float(1.0 - np.isfinite(array).mean()) if len(array) else 1.0


def _resample_source(spec: SourceSpec, config: DataConfig) -> Path:
    output_dir = config.processed_path / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data-00000-of-00001.arrow"
    if output_path.exists():
        existing = _read_arrow(output_path)
        if existing.num_rows == 0 or existing.schema != PROCESSED_SCHEMA:
            raise ValueError(f"existing processed artifact is invalid: {output_path}")
        return output_path

    raw_files = _arrow_files(spec.raw_directories(config.raw_path))
    sink = pa.OSFile(str(output_path), "wb")
    writer = pa.ipc.new_stream(sink, PROCESSED_SCHEMA)
    output_rows = 0
    try:
        for raw_path in raw_files:
            table = _read_arrow(raw_path)
            for row_idx in range(table.num_rows):
                freq = table.column("freq")[row_idx].as_py()
                if freq != spec.expected_freq:
                    raise ValueError(f"{raw_path}: expected {spec.expected_freq}, found {freq}")
                scalar = table.column("target")[row_idx]
                if pa.types.is_list(scalar.values.type):
                    raise ValueError(f"{spec.name} unexpectedly contains multivariate targets")
                values = np.asarray(
                    scalar.values.to_numpy(zero_copy_only=False), dtype=np.float32
                )
                hourly, aligned_start = resample_to_hourly(
                    values,
                    table.column("start")[row_idx].as_py(),
                    freq,
                    spec.min_valid_fraction,
                )
                batch = pa.record_batch(
                    [
                        pa.array([table.column("item_id")[row_idx].as_py()], type=pa.string()),
                        pa.array([aligned_start.replace(tzinfo=None)], type=pa.timestamp("s")),
                        pa.array(["H"], type=pa.string()),
                        pa.array([hourly], type=pa.list_(pa.float32())),
                    ],
                    schema=PROCESSED_SCHEMA,
                )
                writer.write_batch(batch)
                output_rows += 1
    finally:
        writer.close()
        sink.close()
    if output_rows == 0:
        raise ValueError(f"{spec.name} produced no hourly rows")
    return output_path


def _build_manifest(spec: SourceSpec, files: list[Path], config: DataConfig) -> dict:
    source_meta = config.meta_path / spec.name
    source_meta.mkdir(parents=True, exist_ok=True)
    manifest_path = source_meta / "manifest.parquet"
    files_path = source_meta / "files.json"
    report_path = source_meta / "report.json"
    if manifest_path.exists() and files_path.exists() and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if "missing_ratio_scanned" not in report:
            report["missing_ratio_scanned"] = spec.num_variates == 1
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        files_payload = {
            "project_root_relative": True,
            "files": [_relative_path(path, config.project_root) for path in files],
        }
        files_path.write_text(json.dumps(files_payload, indent=2), encoding="utf-8")
        return report

    columns: dict[str, list] = {field.name: [] for field in MANIFEST_SCHEMA}
    expected_freq = "H" if spec.resample else spec.expected_freq
    min_required = min(config.context_lengths) + config.prediction_length

    for file_id, path in enumerate(files):
        table = _read_arrow(path)
        required = {"item_id", "start", "freq", "target"}
        if not required.issubset(table.column_names):
            raise ValueError(f"{path}: invalid schema {table.column_names}")
        for row_idx in range(table.num_rows):
            freq = table.column("freq")[row_idx].as_py()
            if freq != expected_freq:
                raise ValueError(f"{path}: expected hourly data, found {freq}")
            scalar = table.column("target")[row_idx]
            num_variates, length = _series_shape(scalar)
            if num_variates != spec.num_variates:
                raise ValueError(
                    f"{path}:{row_idx} expected {spec.num_variates} variables, found {num_variates}"
                )
            missing_ratio = _missing_ratio(scalar)
            columns["file_id"].append(file_id)
            columns["row_idx"].append(row_idx)
            columns["item_id"].append(table.column("item_id")[row_idx].as_py())
            columns["start"].append(table.column("start")[row_idx].as_py())
            columns["freq"].append(freq)
            columns["length"].append(length)
            columns["num_variates"].append(num_variates)
            columns["missing_ratio"].append(missing_ratio)
            columns["eligible"].append(length >= min_required and missing_ratio < 1.0)

    manifest = pa.Table.from_pydict(columns, schema=MANIFEST_SCHEMA)
    pq.write_table(manifest, manifest_path, compression="zstd")
    files_payload = {
        "project_root_relative": True,
        "files": [_relative_path(path, config.project_root) for path in files],
    }
    files_path.write_text(json.dumps(files_payload, indent=2), encoding="utf-8")

    lengths = np.asarray(columns["length"], dtype=np.int64)
    missing = np.asarray(columns["missing_ratio"], dtype=np.float64)
    eligible = np.asarray(columns["eligible"], dtype=bool)
    report = {
        "source": spec.name,
        "frequency": expected_freq,
        "files": len(files),
        "physical_series": len(lengths),
        "logical_series": int(sum(columns["num_variates"])),
        "eligible_series": int(eligible.sum()),
        "length_min": int(lengths.min()),
        "length_max": int(lengths.max()),
        "missing_ratio_mean": float(missing.mean()),
        "missing_ratio_max": float(missing.max()),
        "missing_ratio_scanned": spec.num_variates == 1,
        "config": {
            "max_context_length": config.max_context_length,
            "prediction_length": config.prediction_length,
            "min_required_length": min_required,
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def preprocess_source(name: str, config: DataConfig) -> dict:
    if name not in SOURCE_TYPES:
        raise KeyError(f"unknown source: {name}")
    spec = SOURCE_TYPES[name].spec
    source_type = SOURCE_TYPES[name]
    if hasattr(source_type, "prepare_processed"):
        files = source_type.prepare_processed(config)
    elif spec.resample:
        files = [_resample_source(spec, config)]
    else:
        files = _arrow_files(spec.raw_directories(config.raw_path))
    return _build_manifest(spec, files, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MeteoTime hourly data manifests")
    parser.add_argument(
        "--source",
        choices=["all", *SOURCE_TYPES.keys()],
        default="all",
        help="source to preprocess",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    config = DataConfig(project_root=args.project_root.resolve())
    config.validate()
    config.meta_path.mkdir(parents=True, exist_ok=True)
    if args.source == "all":
        # Keep the supplementary pressure source next to ERA5 so full
        # preprocessing always builds the pressure manifest as well.
        names = [name for name in SOURCE_TYPES if name != "era5_pressure"]
        if "era5" in names and "era5_pressure" in SOURCE_TYPES:
            names.insert(names.index("era5") + 1, "era5_pressure")
    else:
        names = [args.source]
    reports = [preprocess_source(name, config) for name in names]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
