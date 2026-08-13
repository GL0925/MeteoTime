from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceSpec:
    name: str
    directory_glob: str
    expected_freq: str
    num_variates: int = 1
    resample: bool = False
    min_valid_fraction: float = 0.75

    def raw_directories(self, raw_root: Path) -> list[Path]:
        directories = sorted(path for path in raw_root.glob(self.directory_glob) if path.is_dir())
        if not directories:
            raise FileNotFoundError(f"no directories match {raw_root / self.directory_glob}")
        return directories
