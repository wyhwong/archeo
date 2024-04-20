from typing import Callable

import numpy as np
import pandas as pd

import archeo.logger
import archeo.schemas.common


local_logger = archeo.logger.get_logger(__name__)


NUM_SAMPLES = 500000


def get_mass_func_from_mahapatra(
    mass: archeo.schemas.common.Domain,
    alpha: float = 2.3,
    dm: float = 4.83,
) -> Callable:
    """
    Get a mass function from Mahapatra's mass distribution.

    Args:
    -----
    mass (archeo.schemas.common.Domain):
        Mass domain.

    alpha (float, optional):
        Power law index, by default 2.3.

    dm (float, optional):
        Tapering parameter, by default 4.83.

    Returns:
    -------
    mass_from_mahapatra (Callable):
        Mass function.
    """

    def _f(ds: pd.Series) -> pd.Series:
        """
        Calculate the function f.

        Args:
        -----
        ds : pd.Series
            mass

        Returns
        -----
        f : pd.Series
            Value of the function.
        """

        mp = ds - mass.low
        return np.exp(dm / mp + dm / (mp - dm))

    def smoothing_func(ds: pd.Series) -> pd.Series:
        """
        Smoothing function.

        Args:
        -----
        ds : pd.Series
            mass

        Returns
        -----
        probis : pd.Series
            Probability.
        """

        if mass.low is None or mass.high is None:
            local_logger.error("Both low and high mass must be specified.")
            raise ValueError("Both low and high mass must be specified.")

        probis = ds.copy()
        probis[ds < mass.low + dm] = 1 / (_f(ds[ds < mass.low + dm]) + 1)
        probis[ds > mass.low + dm] = 1
        probis *= ds ** (-alpha)
        return probis

    if mass.low is None or mass.high is None:
        local_logger.error("Both low and high mass must be specified.")
        raise ValueError("Both low and high mass must be specified.")

    masses = pd.Series(np.random.uniform(mass.low, mass.high, NUM_SAMPLES))
    probis = smoothing_func(masses)
    probis /= probis.sum()

    def mass_from_mahapatra() -> float:
        """
        Generate a mass from Mahapatra's mass distribution.

        Returns
        -----
        mass (float):
            Value of mass.
        """

        return np.random.choice(masses, p=probis)

    return mass_from_mahapatra
