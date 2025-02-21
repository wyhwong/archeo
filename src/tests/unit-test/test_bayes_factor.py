import numpy as np
import pandas as pd
import pytest

from archeo.core.bayes import BayesFactorCalculator


np.random.seed(42)
N_SAMPLES = 100000  # Number of samples, the more the better


@pytest.fixture(name="mass_prior")
def default_mass_prior() -> pd.Series:
    """Default mass prior."""

    return pd.Series(np.random.uniform(low=5, high=65, size=N_SAMPLES))


@pytest.fixture(name="mass_posterior")
def default_mass_posterior() -> pd.Series:
    """Default mass posterior."""

    return pd.Series(np.random.normal(loc=35, scale=5, size=N_SAMPLES)).clip(lower=5, upper=65)


def test_bayes_factor_with_no_prior_change(mass_prior, mass_posterior):
    """Test the computation of the Bayes factor."""

    for use_kde in [False, True]:

        calculator = BayesFactorCalculator(use_kde=use_kde)

        bayes_factor, _ = calculator.get_bayes_factor(
            candidate_prior_param=mass_prior,
            posterior_param=mass_posterior,
            prior_param=mass_prior,
        )
        # Here we replace the prior by the original prior
        # The Bayes factor should be exactly 1
        assert np.isclose(bayes_factor, 1, atol=0.02)


def test_bayes_factor_replace_delta_prior(mass_prior, mass_posterior):
    """Test the computation of the Bayes factor."""

    candidate_prior = pd.Series(np.random.normal(loc=35, scale=0.01, size=N_SAMPLES))

    for use_kde in [False, True]:

        calculator = BayesFactorCalculator(nbins=101, use_kde=use_kde)
        bayes_factor, _ = calculator.get_bayes_factor(
            candidate_prior_param=candidate_prior,
            posterior_param=mass_posterior,
            prior_param=mass_prior,
        )
        assert np.isclose(bayes_factor, 4.73944449, atol=0.3)


def test_bayes_factor_replace_flat_normal_prior(mass_prior, mass_posterior):
    """Test the computation of the Bayes factor."""

    candidate_prior = pd.Series(np.random.normal(loc=35, scale=50, size=N_SAMPLES))
    candidate_prior = candidate_prior.loc[candidate_prior.between(5, 65)]

    for use_kde in [False, True]:

        calculator = BayesFactorCalculator(use_kde=use_kde)
        bayes_factor, _ = calculator.get_bayes_factor(
            candidate_prior_param=candidate_prior,
            posterior_param=mass_posterior,
            prior_param=mass_prior,
        )
        assert 1.0 < bayes_factor < 1.1


def test_bayes_factor_replace_flat_beta_prior(mass_prior, mass_posterior):
    """Test the computation of the Bayes factor."""

    candidate_prior = pd.Series(np.random.beta(a=0.92, b=0.92, size=N_SAMPLES))
    candidate_prior = candidate_prior * 60 + 5

    for use_kde in [False, True]:

        calculator = BayesFactorCalculator(use_kde=use_kde)
        bayes_factor, _ = calculator.get_bayes_factor(
            candidate_prior_param=candidate_prior,
            posterior_param=mass_posterior,
            prior_param=mass_prior,
        )
        assert 0.9 < bayes_factor < 1.0
