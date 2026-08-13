from .base import SourceSpec


class ElecDemandSource:
    spec = SourceSpec(
        name="elecdemand",
        directory_glob="elecdemand",
        expected_freq="30T",
        resample=True,
        min_valid_fraction=1.0,
    )
