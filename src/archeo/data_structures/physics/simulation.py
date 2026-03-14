from typing import TypeAlias

import pandas as pd

from archeo.data_structures.physics.binary import Binary, BinaryGenerator
from archeo.data_structures.physics.black_hole import BlackHole


BlackHoleMerger: TypeAlias = tuple[Binary, BlackHole]
BlackHoleMergers: TypeAlias = list[BlackHoleMerger]
PipelineOutput: TypeAlias = tuple[pd.DataFrame, BinaryGenerator]
