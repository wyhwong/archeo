import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from archeo.constants.bayesian import DEFAULT_BINSIZE_MASS, DEFAULT_BINSIZE_SPIN
from archeo.data_structures.math import Domain
from archeo.utils.logger import get_logger


LOGGER = get_logger(__name__)


class ImportanceSamplingDataBase(BaseModel, frozen=True):
    """Importance sampling data"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    posterior_samples: pd.DataFrame
    prior_samples: pd.DataFrame
    new_prior_samples: pd.DataFrame
    binsize_spin: float = DEFAULT_BINSIZE_SPIN
    binsize_mass: float = DEFAULT_BINSIZE_MASS
    ztol: float = 1e-8

    @property
    def common_columns(self) -> list[str]:
        """Return columns shared by posterior, prior, and new-prior dataframes.

        Returns:
            list[str]: Shared column names.
        """

        return sorted(
            list(
                set(self.posterior_samples.columns)
                .intersection(set(self.prior_samples.columns))
                .intersection(set(self.new_prior_samples.columns))
            )
        )

    @property
    def bounds(self) -> dict[str, Domain]:
        """Compute per-column joint bounds across all sample sets.

        Returns:
            dict[str, Domain]: Mapping from column name to value domain.
        """

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

    def get_binsize(self, col_name: str) -> float:
        """Return bin size associated with a column name prefix.

        Args:
            col_name (str): Column name.

        Returns:
            float: Bin size used for histogramming.

        Raises:
            ValueError: If the column prefix is unsupported.
        """

        if col_name.startswith("a"):
            return self.binsize_spin

        if col_name.startswith("m"):
            return self.binsize_mass

        raise ValueError(f"Unknown column name {col_name}")

    def get_nbins(self, col_name: str) -> int:
        """Compute number of histogram bins for a column.

        Args:
            col_name (str): Column name.

        Returns:
            int: Number of bins.
        """

        binsize = self.get_binsize(col_name)
        bounds = self.bounds[col_name]
        return max(1, int(np.ceil((bounds.high - bounds.low) / binsize)))

    def get_edges(self, col_name: str) -> np.ndarray:
        """Compute histogram bin edges for a column.

        Args:
            col_name (str): Column name.

        Returns:
            np.ndarray: Bin edge array.
        """

        nbins = self.get_nbins(col_name)
        bounds = self.bounds[col_name]
        return np.linspace(bounds.low, bounds.high, nbins + 1)

    def get_binwidth(self, col_name: str) -> float:
        """Return scalar histogram bin width for a column.

        Args:
            col_name (str): Column name.

        Returns:
            float: Bin width.
        """

        edges = self.get_edges(col_name)
        return edges[1] - edges[0]

    def _safe_divide(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute stable elementwise ratio with zero-protection.

        Args:
            a (np.ndarray): Numerator values.
            b (np.ndarray): Denominator values.

        Returns:
            np.ndarray: ``a / b`` where ``b`` exceeds tolerance, else 0.
        """

        a_arr = np.asarray(a)
        return np.divide(
            a_arr,
            b,
            out=np.zeros_like(b, dtype=float),
            where=b > self.ztol,
        )
