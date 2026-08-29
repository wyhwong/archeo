import pandas as pd
import pytest

from archeo.preset.simulation.agnostic import simulate_agnostic_aligned_spin_binaries
from archeo.visualization import visualize_posterior_estimation, visualize_prior_distribution


@pytest.fixture(name="prior")
def default_prior():
    """Load the default prior for testing."""

    df_binaries, _ = simulate_agnostic_aligned_spin_binaries(size=1000)

    return df_binaries


@pytest.fixture(name="posterior")
def default_posterior():
    """Load the default posterior for testing."""

    df_binaries, _ = simulate_agnostic_aligned_spin_binaries(size=1000)
    prior = df_binaries

    # Here we inject some NaN samples to test the handling for visualization
    nan_samples = pd.DataFrame(index=range(10), columns=prior.columns)
    posterior = pd.concat([prior, nan_samples], ignore_index=True)

    return posterior


def test_visualizing_posterior_estimation(posterior: pd.DataFrame, tmp_path):
    """Test the visualization of the posterior estimation.

    NOTE:
    - Posterior basically have the same columns as prior,
        so we can use prior as posterior to test the visualization.
    """

    output_dir = tmp_path / "posterior_estimation_test"

    assert not output_dir.exists()

    visualize_posterior_estimation(dfs={"test": posterior}, output_dir=str(output_dir))

    assert output_dir.exists()


def test_visualizing_prior_distribution(prior: pd.DataFrame, tmp_path):
    """Test the visualization of the prior distribution."""

    output_dir = tmp_path / "prior_distribution_test"

    assert not output_dir.exists()

    visualize_prior_distribution(prior, output_dir=str(output_dir))

    assert output_dir.exists()
