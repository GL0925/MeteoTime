from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from .base import SourceSpec

if TYPE_CHECKING:
    from config_data import DataConfig


class ERA5PressureSource:
    """Convert the standalone ERA5 pressure NetCDF files to Arrow series.

    The raw NetCDF files remain untouched. Each grid point becomes one
    two-variate Arrow row, with surface and mean-sea-level pressure as the
    two variables. Time alignment with LOTSA's 364-day Arrow is intentionally
    preserved as metadata rather than silently changing the raw values.
    """

    spec = SourceSpec(
        name="era5_pressure",
        directory_glob="era5_pressure_1989_2018",
        expected_freq="H",
        num_variates=2,
    )

    @staticmethod
    def prepare_processed(config: DataConfig) -> list[Path]:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "ERA5 pressure preprocessing requires h5py; install it in the training environment"
            ) from exc

        raw_dir = config.raw_path / "era5_pressure_1989_2018"
        raw_files = sorted(raw_dir.glob("era5_pressure_*.nc"))
        if not raw_files:
            # Reuse derived Arrow files when the original NetCDF files have
            # been removed after preprocessing.
            arrow_files = sorted(raw_dir.glob("era5_pressure_*.arrow"))
            if arrow_files:
                return arrow_files
            raise FileNotFoundError(f"no ERA5 pressure NetCDF or Arrow files under {raw_dir}")
        # Keep derived Arrow beside the source NetCDFs so the whole dataset is
        # managed as one `era5_pressure` source directory.
        output_dir = raw_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []

        for raw_path in raw_files:
            year = raw_path.stem.split("_")[2]
            output_path = output_dir / f"era5_pressure_{year}.arrow"
            outputs.append(output_path)
            if output_path.exists():
                try:
                    with pa.memory_map(str(output_path), "r") as source:
                        existing = ipc.open_stream(source).read_all()
                    target = existing.column("target")
                    valid = (
                        existing.num_rows == 8192
                        and all(existing.column(name).null_count == 0 for name in existing.column_names)
                        and target[0].as_py() is not None
                        and len(target[0].as_py()) == 2
                        and len(target[0].as_py()[0]) > 0
                    )
                    if valid:
                        continue
                except Exception:
                    pass
            print(f"preparing ERA5 pressure {year}: {raw_path.name}", flush=True)
            with h5py.File(raw_path, "r") as source:
                surface = source["surface_pressure"]
                mean_sea = source["mean_sea_level_pressure"]
                if surface.shape != mean_sea.shape or len(surface.shape) != 3:
                    raise ValueError(f"{raw_path}: pressure variables have incompatible shapes")
                _, lat_count, lon_count = surface.shape
                if (lat_count, lon_count) != (64, 128):
                    raise ValueError(
                        f"{raw_path}: expected 64x128 grid, found {lat_count}x{lon_count}"
                    )
                start = datetime(int(year), 1, 1)
                schema = pa.schema(
                    [
                        ("item_id", pa.string()),
                        ("start", pa.timestamp("s")),
                        ("freq", pa.string()),
                        ("target", pa.list_(pa.list_(pa.float32()), 2)),
                    ]
                )
                with pa.OSFile(str(output_path), "wb") as sink:
                    with pa.ipc.new_stream(sink, schema) as writer:
                        # Read latitude blocks to match the NetCDF chunk layout;
                        # per-grid-point reads cause severe random I/O.
                        for lat_start in range(0, lat_count, 4):
                            lat_end = min(lat_start + 4, lat_count)
                            surface_block = np.asarray(
                                surface[:, lat_start:lat_end, :], dtype=np.float32
                            )
                            mean_sea_block = np.asarray(
                                mean_sea[:, lat_start:lat_end, :], dtype=np.float32
                            )
                            item_ids = []
                            starts = []
                            freqs = []
                            targets = []
                            for lat_offset, lat_index in enumerate(range(lat_start, lat_end)):
                                for lon_index in range(lon_count):
                                    item_ids.append(
                                        f"era5_pressure:{year}_{lat_index}_{lon_index}"
                                    )
                                    starts.append(start)
                                    freqs.append("H")
                                    targets.append(
                                        [
                                            surface_block[:, lat_offset, lon_index].tolist(),
                                            mean_sea_block[:, lat_offset, lon_index].tolist(),
                                        ]
                                    )
                            writer.write_table(
                                pa.Table.from_arrays(
                                    [
                                        pa.array(item_ids, type=pa.string()),
                                        pa.array(starts, type=pa.timestamp("s")),
                                        pa.array(freqs, type=pa.string()),
                                        pa.array(targets, type=schema.field("target").type),
                                    ],
                                    schema=schema,
                                )
                            )
        return outputs
