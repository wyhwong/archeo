import numpy as np
import pandas as pd
import pytest

from archeo.core.forward import ImportanceSamplingData as ISData


np.random.seed(42)
N_SAMPLES = 100000  # Number of samples, the more the better


@pytest.fixture(name="prior")
def default_prior() -> pd.Series:
    """Default mass prior."""

    return pd.DataFrame(
        {
            "m_1": np.random.uniform(low=5, high=65, size=N_SAMPLES),
            "a_1": np.random.uniform(low=0, high=1, size=N_SAMPLES),
        }
    )


@pytest.fixture(name="posterior")
def default_posterior() -> pd.Series:
    """Default mass posterior."""

    df = pd.DataFrame(
        {
            "m_1": np.random.normal(loc=35, scale=5, size=N_SAMPLES),
            "a_1": np.random.uniform(low=0, high=1, size=N_SAMPLES),
        }
    )
    return df.clip(lower={"m_1": 5, "a_1": 0}, upper={"m_1": 65, "a_1": 1})


def test_bayes_factor_with_no_prior_change_1d(prior, posterior):
    """Test the computation of the Bayes factor."""

    candidate_prior = pd.DataFrame({"m_1": np.random.uniform(low=5, high=65, size=N_SAMPLES)})
    data = ISData(
        new_prior_samples=candidate_prior,
        posterior_samples=posterior,
        prior_samples=prior,
        assume_parameter_independence=True,
    )
    bayes_factor = data.get_bayes_factor()

    # Here we replace the prior by the original prior
    # The Bayes factor should be exactly 1
    assert np.isclose(bayes_factor, 1, atol=0.05)


def test_bayes_factor_with_no_prior_change_dd(prior, posterior):
    """Test the computation of the Bayes factor."""

    candidate_prior = pd.DataFrame({"m_1": np.random.uniform(low=5, high=65, size=N_SAMPLES)})
    data = ISData(
        new_prior_samples=candidate_prior,
        posterior_samples=posterior,
        prior_samples=prior,
        assume_parameter_independence=False,
    )
    bayes_factor = data.get_bayes_factor()

    # Here we replace the prior by the original prior
    # The Bayes factor should be exactly 1
    assert np.isclose(bayes_factor, 1, atol=0.05)


def test_bayes_factor_replace_delta_prior_1d(prior, posterior):
    """Test the computation of the Bayes factor."""

    candidate_prior = pd.DataFrame({"m_1": np.random.normal(loc=35, scale=0.01, size=N_SAMPLES)})
    data = ISData(
        new_prior_samples=candidate_prior,
        posterior_samples=posterior,
        prior_samples=prior,
        assume_parameter_independence=True,
    )
    bayes_factor = data.get_bayes_factor()

    # Expected Bayes factor is 4.73944449
    assert np.isclose(bayes_factor, 4.73944449, atol=0.5)


def test_bayes_factor_replace_delta_prior_dd(prior, posterior):
    """Test the computation of the Bayes factor."""

    candidate_prior = pd.DataFrame({"m_1": np.random.normal(loc=35, scale=0.01, size=N_SAMPLES)})
    data = ISData(
        new_prior_samples=candidate_prior,
        posterior_samples=posterior,
        prior_samples=prior,
        assume_parameter_independence=False,
    )
    bayes_factor = data.get_bayes_factor()

    # Expected Bayes factor is 4.73944449
    assert np.isclose(bayes_factor, 4.73944449, atol=0.5)


def test_bayes_factor_replace_flat_normal_prior_1d(prior, posterior):
    """Test the computation of the Bayes factor."""

    samples = np.random.normal(loc=35, scale=50, size=N_SAMPLES)
    samples = samples[(5 <= samples) & (samples <= 65)]
    candidate_prior = pd.DataFrame({"m_1": samples})
    data = ISData(
        new_prior_samples=candidate_prior,
        posterior_samples=posterior,
        prior_samples=prior,
        assume_parameter_independence=True,
    )
    bayes_factor = data.get_bayes_factor()

    assert 1.0 <= bayes_factor <= 1.1


def test_bayes_factor_replace_flat_normal_prior_dd(prior, posterior):
    """Test the computation of the Bayes factor."""

    samples = np.random.normal(loc=35, scale=50, size=N_SAMPLES)
    samples = samples[(5 <= samples) & (samples <= 65)]
    candidate_prior = pd.DataFrame({"m_1": samples})
    data = ISData(
        new_prior_samples=candidate_prior,
        posterior_samples=posterior,
        prior_samples=prior,
        assume_parameter_independence=False,
    )
    bayes_factor = data.get_bayes_factor()

    assert 1.0 <= bayes_factor <= 1.1


def test_bayes_factor_replace_flat_beta_prior_1d(prior, posterior):
    """Test the computation of the Bayes factor."""

    candidate_prior = pd.DataFrame({"m_1": np.random.beta(a=0.92, b=0.92, size=N_SAMPLES) * 60 + 5})
    data = ISData(
        new_prior_samples=candidate_prior,
        posterior_samples=posterior,
        prior_samples=prior,
        assume_parameter_independence=True,
    )
    bayes_factor = data.get_bayes_factor()

    assert 0.9 <= bayes_factor <= 1.0


def test_bayes_factor_replace_flat_beta_prior_dd(prior, posterior):
    """Test the computation of the Bayes factor."""

    candidate_prior = pd.DataFrame({"m_1": np.random.beta(a=0.92, b=0.92, size=N_SAMPLES) * 60 + 5})
    data = ISData(
        new_prior_samples=candidate_prior,
        posterior_samples=posterior,
        prior_samples=prior,
        assume_parameter_independence=False,
    )
    bayes_factor = data.get_bayes_factor()

    assert 0.9 <= bayes_factor <= 1.0
