from .base import SourceSpec


class ERA5Source:
    spec = SourceSpec(
        name="era5",
        directory_glob="era5_????",
        expected_freq="H",
        num_variates=45,
    )
