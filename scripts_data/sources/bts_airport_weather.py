from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .base import SourceSpec

if TYPE_CHECKING:
    from config_data import DataConfig


FEATURES = {
    "temp_c": (-100.0, 70.0),
    "relative_humidity_pct": (0.0, 100.0),
    "wind_speed_ms": (0.0, 150.0),
    "sea_level_pressure_hpa": (800.0, 1100.0),
}


class BTSAirportWeatherSource:
    """Convert the BTS airport observation panel into univariate hourly series."""

    spec = SourceSpec(
        name="bts_airport_weather",
        directory_glob="bts_flights_weather",
        expected_freq="H",
    )

    @staticmethod
    def prepare_processed(config: DataConfig) -> list[Path]:
        # Local import avoids a preprocessing/source import cycle.
        from scripts_data.preprocess import PROCESSED_SCHEMA, _read_arrow

        output_dir = config.processed_path / BTSAirportWeatherSource.spec.name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "data-00000-of-00001.arrow"
        if output_path.exists():
            existing = _read_arrow(output_path)
            if existing.num_rows == 0 or existing.schema != PROCESSED_SCHEMA:
                raise ValueError(f"existing processed artifact is invalid: {output_path}")
            return [output_path]

        raw_dir = BTSAirportWeatherSource.spec.raw_directories(config.raw_path)[0]
        input_path = raw_dir / "airport_hourly_weather.parquet"
        if not input_path.exists():
            raise FileNotFoundError(f"BTS weather panel is missing: {input_path}")

        columns = [
            "isd_station_id",
            "observation_hour_utc",
            "obs_missing",
            "obs_is_imputed",
            *FEATURES,
        ]
        parquet = pq.ParquetFile(input_path)
        sink = pa.OSFile(str(output_path), "wb")
        writer = pa.ipc.new_stream(sink, PROCESSED_SCHEMA)
        current_station: str | None = None
        chunks: list[dict[str, np.ndarray]] = []

        def flush_station() -> None:
            nonlocal chunks
            if not chunks:
                return
            timestamps = np.concatenate([chunk["timestamp"] for chunk in chunks])
            if np.any(np.diff(timestamps) <= 0):
                raise ValueError(f"BTS station {current_station} is not strictly time-ordered")
            start_epoch = int(timestamps[0])
            hour_index = (timestamps - start_epoch) // 3600
            length = int(hour_index[-1]) + 1
            start = datetime.fromtimestamp(start_epoch, tz=timezone.utc).replace(tzinfo=None)
            for feature, (lower, upper) in FEATURES.items():
                series = np.full(length, np.nan, dtype=np.float32)
                values = np.concatenate([chunk[feature] for chunk in chunks]).astype(
                    np.float32, copy=False
                )
                valid = np.isfinite(values) & (values >= lower) & (values <= upper)
                series[hour_index[valid]] = values[valid]
                writer.write_batch(
                    pa.record_batch(
                        [
                            pa.array([f"bts:{current_station}:{feature}"], type=pa.string()),
                            pa.array([start], type=pa.timestamp("s")),
                            pa.array(["H"], type=pa.string()),
                            pa.array([series], type=pa.list_(pa.float32())),
                        ],
                        schema=PROCESSED_SCHEMA,
                    )
                )
            chunks = []

        try:
            for row_group in range(parquet.num_row_groups):
                table = parquet.read_row_group(row_group, columns=columns)
                station_ids = np.asarray(table["isd_station_id"].to_pylist(), dtype=object)
                timestamps = (
                    table["observation_hour_utc"].cast(pa.int64()).to_numpy(zero_copy_only=False)
                    // 1_000_000
                ).astype(np.int64, copy=False)
                missing = np.asarray(table["obs_missing"].to_pylist(), dtype=bool)
                imputed = np.asarray(table["obs_is_imputed"].to_pylist(), dtype=bool)
                values = {
                    feature: np.asarray(table[feature].to_pylist(), dtype=np.float32)
                    for feature in FEATURES
                }
                invalid = missing | imputed
                for feature in FEATURES:
                    values[feature][invalid] = np.nan

                boundaries = np.r_[0, np.flatnonzero(station_ids[1:] != station_ids[:-1]) + 1, len(table)]
                for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
                    station = str(station_ids[start_idx])
                    if current_station is not None and station != current_station:
                        flush_station()
                    current_station = station
                    chunks.append(
                        {
                            "timestamp": timestamps[start_idx:end_idx],
                            **{
                                feature: values[feature][start_idx:end_idx]
                                for feature in FEATURES
                            },
                        }
                    )
            flush_station()
        finally:
            writer.close()
            sink.close()
        return [output_path]
