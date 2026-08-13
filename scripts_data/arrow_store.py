from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa


class ArrowShardCache:
    """Process-local mmap cache. Create one cache inside each DataLoader worker."""

    def __init__(self, max_open_files: int = 12) -> None:
        if max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        self.max_open_files = max_open_files
        self._entries: OrderedDict[Path, tuple[pa.MemoryMappedFile, pa.Table]] = OrderedDict()

    def table(self, path: str | Path) -> pa.Table:
        path = Path(path).resolve()
        cached = self._entries.pop(path, None)
        if cached is not None:
            self._entries[path] = cached
            return cached[1]

        source = pa.memory_map(str(path), "r")
        table = pa.ipc.open_stream(source).read_all()
        self._entries[path] = (source, table)
        while len(self._entries) > self.max_open_files:
            _, (old_source, _) = self._entries.popitem(last=False)
            old_source.close()
        return table

    def read_series(self, path: str | Path, row_idx: int, variable_idx: int = 0) -> np.ndarray:
        scalar = self.table(path).column("target")[row_idx]
        values = scalar.values
        if pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
            if not 0 <= variable_idx < len(values):
                raise IndexError(f"variable_idx {variable_idx} out of range")
            values = values[variable_idx].values
        elif variable_idx != 0:
            raise IndexError("univariate series only supports variable_idx=0")
        return np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32).copy()

    def close(self) -> None:
        while self._entries:
            _, (source, _) = self._entries.popitem(last=False)
            source.close()

    def __del__(self) -> None:
        self.close()
