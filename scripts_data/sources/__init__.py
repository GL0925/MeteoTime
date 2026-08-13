from .bts_airport_weather import BTSAirportWeatherSource
from .elecdemand import ElecDemandSource
from .era5 import ERA5Source
from .era5_pressure import ERA5PressureSource
from .lcl import LCLSource
from .oikolab import OikolabSource
from .wind_farms import WindFarmsSource
from .wind_power import WindPowerSource

SOURCE_TYPES = {
    "bts_airport_weather": BTSAirportWeatherSource,
    "era5": ERA5Source,
    "era5_pressure": ERA5PressureSource,
    "oikolab_weather": OikolabSource,
    "lcl": LCLSource,
    "wind_farms_with_missing": WindFarmsSource,
    "elecdemand": ElecDemandSource,
    "wind_power": WindPowerSource,
}

__all__ = ["SOURCE_TYPES"]
