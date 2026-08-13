from .base import SourceSpec


class LCLSource:
    spec = SourceSpec(name="lcl", directory_glob="lcl", expected_freq="H")
