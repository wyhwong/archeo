from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, PositiveFloat

from archeo.data_structures.distribution import DistributionBase
from archeo.data_structures.math import Domain


class MahapatraMassFunction(BaseModel, DistributionBase, frozen=True):
    """Get a mass function from Mahapatra's mass distribution.
    NOTE: For details, see https://arxiv.org/abs/2209.05766.

    Args:
        mass (Domain): Mass domain.
        alpha (float): Power law index.
        dm (float): Tapering parameter.
        resolution (float): Resolution of the mass function.
    """

    mass: Domain
    alpha: PositiveFloat = 2.3
    dm: PositiveFloat = 4.83
    resolution: PositiveFloat = 0.001

    @property
    def masses(self) -> np.ndarray:
        """Masses to evaluate the mass function on."""

        return np.arange(self.mass.low, self.mass.high + self.resolution, self.resolution)

    @property
    def probis(self) -> np.ndarray:
        """Probabilities of the masses."""

        probis = self._smoothing_func(self.masses)
        probis /= probis.sum()
        return probis

    def _f(self, masses: np.ndarray) -> np.ndarray:
        """Calculate the function f in Mahapatra's paper"""

        mp = masses - self.mass.low
        return np.exp(self.dm / mp + self.dm / (mp - self.dm))

    def _smoothing_func(self, masses: np.ndarray) -> np.ndarray:
        """Smoothing function."""

        probis = masses.copy()
        probis[masses < self.mass.low + self.dm] = 1 / (self._f(masses[masses < self.mass.low + self.dm]) + 1)
        probis[masses > self.mass.low + self.dm] = 1
        probis *= masses ** (-self.alpha)
        return probis

    @property
    def min(self) -> float:
        """Minimum value of the distribution."""

        return self.mass.low

    @property
    def max(self) -> float:
        """Maximum value of the distribution."""

        return self.mass.high

    def draw(self, size: Optional[int] = None) -> Union[float, np.ndarray[float]]:
        """Draw samples from Mahapatra's mass distribution."""

        return np.random.choice(self.masses, size=size, p=self.probis)
