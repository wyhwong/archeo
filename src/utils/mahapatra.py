import numpy as np
import pandas as pd
from typing import Callable

import schemas.common

NUM_SAMPLES = 500000


def get_mass_func_from_mahapatra(mass: schemas.common.Domain, alpha=2.3, dm=4.83) -> Callable:
    """
    Get a mass function from Mahapatra's mass distribution.

    Parameters
    ----------
    mass : schemas.common.Domain
        Mass domain.
    alpha : float, optional
        Power law index, by default 2.3.
    dm : float, optional
        Tapering parameter, by default 4.83.

    Returns
    -------
    mass_from_mahapatra : Callable
        Mass function.
    """

    def _f(ds: pd.Series) -> pd.Series:
        """
        Calculate the function f.

        Parameters
        ----------
        ds : pd.Series
            mass

        Returns
        -------
        f : pd.Series
            Value of the function.
        """
        mp = ds - mass.low
        return np.exp(dm / mp + dm / (mp - dm))

    def smoothing_func(ds: pd.Series) -> pd.Series:
        """
        Smoothing function.

        Parameters
        ----------
        ds : pd.Series
            mass

        Returns
        -------
        probis : pd.Series
            Probability.
        """
        probis = ds.copy()
        probis[ds < mass.low + dm] = 1 / (_f(ds[ds < mass.low + dm]) + 1)
        probis[ds > mass.low + dm] = 1
        probis *= ds ** (-alpha)
        return probis

    masses = pd.Series(np.random.uniform(mass.low, mass.high, NUM_SAMPLES))
    probis = smoothing_func(masses)
    probis /= probis.sum()

    def mass_from_mahapatra():
        """
        Generate a mass from Mahapatra's mass distribution.

        Returns
        -------
        mass : float
            Value of mass.
        """
        return np.random.choice(masses, p=probis)

    return mass_from_mahapatra
