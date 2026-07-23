from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, PositiveFloat

from archeo.constants.physics import BH_MASS_LB, PISN_LB
from archeo.data_structures.distribution import DistributionBase
from archeo.data_structures.math import Domain


class MahapatraMassFunction(BaseModel, DistributionBase, frozen=True):
    """Mahapatra-smoothed power-law mass distribution.

    This distribution implements a power-law profile modulated by a smoothing
    function near the lower-mass edge, following the Mahapatra-style prescription.
    Samples are generated over a discretized mass grid and normalized to form a
    valid probability mass function. For details, see https://arxiv.org/abs/2209.05766.

    Args:
        mass (Domain): Allowed mass interval.
        alpha (float): Power-law index.
        dm (float): Smoothing scale near the low-mass cutoff.
        resolution (float): Grid spacing used for numerical evaluation.
    """

    mass: Domain = Domain(low=BH_MASS_LB, high=PISN_LB)
    alpha: PositiveFloat = 2.3
    dm: PositiveFloat = 4.83
    resolution: PositiveFloat = 0.001

    @property
    def masses(self) -> np.ndarray:
        """Return the grid of mass values used to evaluate the distribution.

        Returns:
            np.ndarray: One-dimensional array spanning `[mass.low, mass.high]` with
            step size `resolution`.
        """

        return np.arange(self.mass.low, self.mass.high + self.resolution, self.resolution)

    @property
    def probis(self) -> np.ndarray:
        """Return normalized probabilities over the mass grid.

        Returns:
            np.ndarray: Probability mass function aligned with `self.masses`.
        """

        probis = self._smoothing_func(self.masses)
        probis /= probis.sum()
        return probis

    def _f(self, masses: np.ndarray) -> np.ndarray:
        """Compute the auxiliary tapering function from Mahapatra et al.

        Args:
            masses (np.ndarray): Mass values where the function is evaluated.

        Returns:
            np.ndarray: Function values used by the smoothing term.
        """

        mp = masses - self.mass.low
        out = np.full_like(masses, np.inf, dtype=float)

        mask = (mp > 0) & (mp < self.dm)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            out[mask] = np.exp(self.dm / mp[mask] + self.dm / (mp[mask] - self.dm))

        return out

    def _smoothing_func(self, masses: np.ndarray) -> np.ndarray:
        """Compute the smoothed power-law profile over input masses.

        Args:
            masses (np.ndarray): Mass values where the profile is evaluated.

        Returns:
            np.ndarray: Unnormalized probabilities before global normalization.
        """

        probis = masses.copy()
        probis[masses < self.mass.low + self.dm] = 1 / (self._f(masses[masses < self.mass.low + self.dm]) + 1)
        probis[masses > self.mass.low + self.dm] = 1
        probis *= masses ** (-self.alpha)
        return probis

    @property
    def min(self) -> float:
        """Return the lower support bound of the distribution.

        Returns:
            float: Lower bound.
        """

        return self.mass.low

    @property
    def max(self) -> float:
        """Return the upper support bound of the distribution.

        Returns:
            float: Upper bound.
        """

        return self.mass.high

    def draw(self, size: Optional[int] = None, random_state: Optional[int] = None) -> Union[float, np.ndarray[float]]:
        """Draw random samples from the Mahapatra mass distribution.

        Args:
            size (Optional[int]): Number of samples to draw. If `None`, returns one sample.
            random_state (Optional[int]): Random seed for reproducibility.

        Returns:
            Union[float, np.ndarray[float]]: Scalar sample or sample array.
        """

        rng = np.random.default_rng(random_state)
        return rng.choice(self.masses, size=size, p=self.probis)
