from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import archeo.logger
from archeo.schema import Domain


local_logger = archeo.logger.get_logger(__name__)


@dataclass
class ImportanceSamplingDataBase:
    """Importance sampling data"""

    posterior_samples: pd.DataFrame
    prior_samples: pd.DataFrame
    new_prior_samples: pd.DataFrame
    binsize_spin: float = 0.05
    binsize_mass: float = 1.0

    def __post_init__(self) -> None:
        """Compute the bounds of the dataframes"""

        self._bounds = self._compute_bounds()

    @property
    def common_columns(self) -> list[str]:
        """Get the common columns between posterior and prior samples"""

        return list(
            set(self.posterior_samples.columns)
            .intersection(set(self.prior_samples.columns))
            .intersection(set(self.new_prior_samples.columns))
        )

    @property
    def bounds(self) -> dict[str, Domain]:
        """Get the bounds of the dataframes"""

        return self._bounds

    def get_binsize(self, col_name: str) -> Optional[float]:

        if col_name.startswith("a"):
            return self.binsize_spin

        if col_name.startswith("m"):
            return self.binsize_mass

        raise ValueError(f"Unknown column name {col_name}")

    def get_nbins(self, col_name: str) -> Optional[int]:
        """Get the number of bins for a given column name"""

        binsize = self.get_binsize(col_name)
        bounds = self.bounds[col_name]
        return int((bounds.high - bounds.low) / binsize)

    def _compute_bounds(self) -> dict[str, Domain]:
        """Get the bounds of the dataframes"""

        bounds: dict[str, Domain] = {}

        for col in self.common_columns:
            _min = min(
                self.posterior_samples[col].min(),
                self.prior_samples[col].min(),
                self.new_prior_samples[col].min(),
            )
            _max = max(
                self.posterior_samples[col].max(),
                self.prior_samples[col].max(),
                self.new_prior_samples[col].max(),
            )
            bounds[col] = Domain(low=_min, high=_max)

        return bounds

    def get_edges(self, col_name: str) -> np.ndarray:
        """Get the edges of the histogram for a given column name"""

        nbins = self.get_nbins(col_name)
        bounds = self.bounds[col_name]
        return np.linspace(bounds.low, bounds.high, nbins + 1)

    def get_binwidth(self, col_name: str) -> float:
        """Get the bin width for a given column name"""

        edges = self.get_edges(col_name)
        return edges[1] - edges[0]

    @staticmethod
    def _safe_divide(a: np.ndarray, b: np.ndarray, ztol: float = 1e-8) -> np.ndarray:
        """Safe division to avoid division by zero"""

        return np.where(b > ztol, np.exp(np.log(a) - np.log(b)), 0.0)
