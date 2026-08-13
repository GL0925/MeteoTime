from __future__ import annotations

from datetime import datetime, timezone

import numpy as np


FREQUENCY_SECONDS = {
    "H": 3600,
    "h": 3600,
    "T": 60,
    "min": 60,
    "30T": 1800,
    "30min": 1800,
    "4S": 4,
    "4s": 4,
}


def _epoch_seconds(start: datetime | np.datetime64) -> int:
    if isinstance(start, np.datetime64):
        return int(start.astype("datetime64[s]").astype(np.int64))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return int(start.timestamp())


def resample_to_hourly(
    values: np.ndarray,
    start: datetime | np.datetime64,
    freq: str,
    min_valid_fraction: float = 0.75,
) -> tuple[np.ndarray, datetime]:
    """Aggregate regular sub-hourly observations into complete UTC clock hours."""
    if freq not in FREQUENCY_SECONDS:
        raise ValueError(f"unsupported frequency: {freq}")
    step_seconds = FREQUENCY_SECONDS[freq]
    if step_seconds == 3600:
        clean = np.asarray(values, dtype=np.float32).copy()
        clean[~np.isfinite(clean)] = np.nan
        return clean, datetime.fromtimestamp(_epoch_seconds(start), tz=timezone.utc)
    if 3600 % step_seconds:
        raise ValueError(f"frequency {freq} does not divide one hour")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1]")

    epoch = _epoch_seconds(start)
    points_per_hour = 3600 // step_seconds
    offset_in_hour = epoch % 3600
    first_bin_count = min(
        len(values), (3600 - offset_in_hour + step_seconds - 1) // step_seconds
    )
    if first_bin_count == points_per_hour:
        start_idx = 0
        aligned_epoch = epoch - offset_in_hour
    else:
        start_idx = first_bin_count
        aligned_epoch = epoch - offset_in_hour + 3600
    num_hours = (len(values) - start_idx) // points_per_hour
    if num_hours <= 0:
        return np.empty(0, dtype=np.float32), datetime.fromtimestamp(
            aligned_epoch, tz=timezone.utc
        )

    usable = np.asarray(values, dtype=np.float32)[
        start_idx : start_idx + num_hours * points_per_hour
    ].reshape(num_hours, points_per_hour)
    valid = np.isfinite(usable)
    counts = valid.sum(axis=1)
    sums = np.where(valid, usable, 0.0).sum(axis=1, dtype=np.float64)
    hourly = np.full(num_hours, np.nan, dtype=np.float32)
    minimum_count = int(np.ceil(points_per_hour * min_valid_fraction))
    accepted = counts >= minimum_count
    hourly[accepted] = (sums[accepted] / counts[accepted]).astype(np.float32)
    aligned_start = datetime.fromtimestamp(aligned_epoch, tz=timezone.utc)
    return hourly, aligned_start
