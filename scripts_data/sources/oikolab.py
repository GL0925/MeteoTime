from .base import SourceSpec


class OikolabSource:
    spec = SourceSpec(
        name="oikolab_weather",
        directory_glob="oikolab_weather",
        expected_freq="H",
    )
