from types import MappingProxyType
from typing import Mapping

from moving_det.vrud.types import SequenceKey


PILOT_SPLITS: Mapping[str, tuple[SequenceKey, ...]] = MappingProxyType(
    {
        "train": (
            SequenceKey("site19", "DJI_20240919154443_0005_V"),
            SequenceKey("site19", "DJI_20240919162906_0003_V"),
            SequenceKey("site22", "DJI_20240719181132_0001_V"),
            SequenceKey("site22", "DJI_20240719091331_0001_V"),
            SequenceKey("site22", "DJI_20240719181521_0002_V"),
            SequenceKey("site22", "DJI_20240719085001_0003_V"),
        ),
        "validation": (
            SequenceKey("site19", "DJI_20240919150818_0004_V"),
            SequenceKey("site22", "DJI_20240719171610_0003_V"),
            SequenceKey("site22", "DJI_20240719085350_0004_V"),
        ),
        "test": (
            SequenceKey("site19", "DJI_20240919093341_0002_V"),
            SequenceKey("site22", "DJI_20240719224127_0006_V"),
            SequenceKey("site22", "DJI_20240719183036_0006_V"),
        ),
    }
)
