from typing import Callable

import numpy as np
import pandas as pd

import archeo.logger
from archeo.schema import Domain


local_logger = archeo.logger.get_logger(__name__)


def get_mahapatra_mass_fn(mass: Domain, alpha=2.3, dm=4.83, n_samples=500000) -> Callable:
    """Get a mass function from Mahapatra's mass distribution.
    NOTE: For details, see https://arxiv.org/abs/2209.05766.

    Args:
        mass (Domain): Mass domain.
        alpha (float): Power law index.
        dm (float): Tapering parameter.
        n_samples (int): Number of samples to generate.

    Returns:
        mass_fn (Callable): Mass function.
    """

    def _f(ds: pd.Series) -> pd.Series:
        """Calculate the function f in Mahapatra's paper"""

        mp = ds - mass.low
        return np.exp(dm / mp + dm / (mp - dm))

    def smoothing_func(ds: pd.Series) -> pd.Series:
        """Smoothing function."""

        probis = ds.copy()
        probis[ds < mass.low + dm] = 1 / (_f(ds[ds < mass.low + dm]) + 1)  # type: ignore
        probis[ds > mass.low + dm] = 1  # type: ignore
        probis *= ds ** (-alpha)
        return probis

    masses = pd.Series(np.random.uniform(mass.low, mass.high, size=n_samples))
    probis = smoothing_func(masses)
    probis /= probis.sum()

    def mass_from_mahapatra() -> float:
        """Generate a mass from Mahapatra's mass distribution."""

        return np.random.choice(masses, p=probis)

    return mass_from_mahapatra
