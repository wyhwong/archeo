import numpy as np
import pandas as pd
import pytest

from archeo.core.forward.bayes import ISData, get_bayes_factor


np.random.seed(42)
N_SAMPLES = 100000  # Number of samples, the more the better


@pytest.fixture(name="prior")
def default_prior() -> pd.Series:
    """Default mass prior."""

    return pd.DataFrame(
        {
            "mass": np.random.uniform(low=5, high=65, size=N_SAMPLES),
            "spin": np.random.uniform(low=0, high=1, size=N_SAMPLES),
        }
    )


@pytest.fixture(name="posterior")
def default_posterior() -> pd.Series:
    """Default mass posterior."""

    df = pd.DataFrame(
        {
            "mass": np.random.normal(loc=35, scale=5, size=N_SAMPLES),
            "spin": np.random.uniform(low=0, high=1, size=N_SAMPLES),
        }
    )
    return df.clip(lower={"mass": 5, "spin": 0}, upper={"mass": 65, "spin": 1})


def test_bayes_factor_with_no_prior_change(prior, posterior):
    """Test the computation of the Bayes factor."""

    candidate_prior = {"mass": np.random.uniform(low=5, high=65, size=N_SAMPLES)}
    data = ISData(candidate_prior=candidate_prior, posterior=posterior, prior=prior)
    bayes_factor = get_bayes_factor(data, random_state=42)

    # Here we replace the prior by the original prior
    # The Bayes factor should be exactly 1
    assert np.isclose(bayes_factor, 1, atol=0.05)


def test_bayes_factor_replace_delta_prior(prior, posterior):
    """Test the computation of the Bayes factor."""

    candidate_prior = {"mass": np.random.normal(loc=35, scale=0.01, size=N_SAMPLES)}
    data = ISData(candidate_prior=candidate_prior, posterior=posterior, prior=prior)
    bayes_factor = get_bayes_factor(data, random_state=42)

    # Expected Bayes factor is 4.73944449
    assert np.isclose(bayes_factor, 4.73944449, atol=0.5)


def test_bayes_factor_replace_flat_normal_prior(prior, posterior):
    """Test the computation of the Bayes factor."""

    samples = np.random.normal(loc=35, scale=50, size=N_SAMPLES)
    samples = samples[(5 <= samples) & (samples <= 65)]
    candidate_prior = {"mass": samples}
    data = ISData(candidate_prior=candidate_prior, posterior=posterior, prior=prior)
    bayes_factor = get_bayes_factor(data, random_state=42)

    assert 1.0 <= bayes_factor <= 1.1


def test_bayes_factor_replace_flat_beta_prior(prior, posterior):
    """Test the computation of the Bayes factor."""

    candidate_prior = {"mass": np.random.beta(a=0.92, b=0.92, size=N_SAMPLES) * 60 + 5}
    data = ISData(candidate_prior=candidate_prior, posterior=posterior, prior=prior)
    bayes_factor = get_bayes_factor(data, random_state=42)

    assert 0.9 <= bayes_factor <= 1.0
