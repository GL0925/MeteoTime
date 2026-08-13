from .base import SourceSpec


class WindFarmsSource:
    spec = SourceSpec(
        name="wind_farms_with_missing",
        directory_glob="wind_farms_with_missing",
        expected_freq="T",
        resample=True,
        min_valid_fraction=0.75,
    )
