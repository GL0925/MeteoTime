from .base import SourceSpec


class WindPowerSource:
    spec = SourceSpec(
        name="wind_power",
        directory_glob="wind_power",
        expected_freq="4S",
        resample=True,
        min_valid_fraction=0.75,
    )
