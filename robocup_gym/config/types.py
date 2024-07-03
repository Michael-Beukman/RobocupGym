from typing import Any
import numpy as np


# Some general types
Vec3 = tuple[float, float, float]
ObservationType = np.ndarray
ActionType = np.ndarray
InfoType = dict[str, Any]

MaybeInfoType = InfoType | None
SeedType = int | None
