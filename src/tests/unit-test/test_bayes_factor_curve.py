import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from archeo.constants import Columns as C
from archeo.core.bayes import BayesFactorCalculator


np.random.seed(42)
# Number of samples, smaller number is fine, here we just ensure the visualization works
N_SAMPLES = 1000


@pytest.fixture(name="mass_prior")
def default_mass_prior() -> pd.Series:
    """Default mass prior."""

    return pd.Series(np.random.uniform(low=5, high=65, size=N_SAMPLES))


@pytest.fixture(name="mass_posterior")
def default_mass_posterior() -> pd.Series:
    """Default mass posterior."""

    return pd.Series(np.random.normal(loc=35, scale=5, size=N_SAMPLES)).clip(lower=5, upper=65)


@pytest.fixture(name="spin_prior")
def default_spin_prior() -> pd.Series:
    """Default mass prior."""

    return pd.Series(np.random.uniform(low=0, high=1, size=N_SAMPLES))


@pytest.fixture(name="spin_posterior")
def default_spin_posterior() -> pd.Series:
    """Default mass posterior."""

    return pd.Series(np.random.normal(loc=0.5, scale=0.1, size=N_SAMPLES)).clip(lower=0, upper=1.0)


@pytest.fixture(name="remnant_prior")
def default_remnant_prior() -> pd.Series:
    """Default remnant prior."""

    return pd.DataFrame(
        data={
            C.BH_MASS: np.random.uniform(low=5, high=65, size=N_SAMPLES),
            C.BH_SPIN: np.random.uniform(low=0, high=1, size=N_SAMPLES),
            C.BH_KICK: np.random.uniform(low=0, high=500, size=N_SAMPLES),
        }
    )


def test_bayes_factor_curve(
    mass_prior: pd.Series,
    mass_posterior: pd.Series,
    spin_prior: pd.Series,
    spin_posterior: pd.Series,
    remnant_prior: pd.DataFrame,
):
    """Test the computation of the Bayes factor curve."""

    for use_kde in [False, True]:
        _, ax = plt.subplots()

        calculator = BayesFactorCalculator(nbins=10, use_kde=use_kde)
        calculator.plot_bayes_factor_over_kick(
            ax=ax,
            label="testing",
            data_bh1={
                "mass_prior": mass_prior,
                "mass_posterior": mass_posterior,
                "spin_prior": spin_prior,
                "spin_posterior": spin_posterior,
                "candidate_prior": remnant_prior,
            },
            data_bh2={
                "mass_prior": mass_prior,
                "mass_posterior": mass_posterior,
                "spin_prior": spin_prior,
                "spin_posterior": spin_posterior,
                "candidate_prior": remnant_prior,
            },
            least_n_samples=100,
        )

        plt.close()
