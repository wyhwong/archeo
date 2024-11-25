import os
from shutil import rmtree

import pytest

import archeo
from archeo.visualization import visualize_posterior_estimation, visualize_prior_distribution


@pytest.fixture(name="prior")
def default_prior():
    """Load the default prior for testing."""

    filepath = f"{os.path.dirname(os.path.dirname(__file__))}/test_data/prior.json"

    return archeo.Prior.from_json(filepath)


@pytest.fixture(name="output_dir")
def default_output_dir():
    """Get the default output directory for testing."""

    return f"{os.path.dirname(os.path.dirname(__file__))}/test_data/visualization"


def test_visualizing_posterior_estimation(prior, output_dir):
    """Test the visualization of the posterior estimation.

    NOTE:
    - Posterior basically have the same columns as prior,
        so we can use prior as posterior to test the visualization.
    - After test, we will remove the output directory.
    """

    assert not os.path.exists(output_dir)

    visualize_posterior_estimation(dfs={"test": prior}, output_dir=output_dir)

    assert os.path.exists(output_dir)

    # Clean up all the files and directories created during the test
    rmtree(output_dir)


def test_visualizing_prior_distribution(prior, output_dir):
    """Test the visualization of the prior distribution.

    NOTE:
    - After test, we will remove the output directory.
    """

    assert not os.path.exists(output_dir)

    visualize_prior_distribution(prior, output_dir=output_dir)

    assert os.path.exists(output_dir)

    # Clean up all the files and directories created during the test
    rmtree(output_dir)
