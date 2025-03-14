import os

import pytest

import archeo
from archeo.constants import Columns as C


@pytest.fixture(name="prior")
def default_prior():
    """Load the default prior for testing."""

    filepath = f"{os.path.dirname(os.path.dirname(__file__))}/test_data/prior.json"

    return archeo.Prior.from_json(filepath)


def test_self_estimate(prior):
    """Run an estimation on the test dataset,
    where we let prior=posterior. Suppose we should get 100% recovery rate.
    """

    mass_posterior = prior[C.BH_MASS].copy()
    spin_posterior = prior[C.BH_SPIN].copy()

    # Run the estimation on itself
    posterior = prior.to_posterior(mass_posterior, spin_posterior)
    assert posterior[C.RECOVERY_RATE].iloc[0] == 1.0

    # Rescale the masses to have m_min' > m_max + mass tolerance
    # And rerun the estimation, suppose recovery rate is 0
    mass_posterior += mass_posterior.max() + prior._mass_tolerance  # pylint: disable=protected-access
    posterior = prior.to_posterior(mass_posterior, spin_posterior)
    assert posterior[C.RECOVERY_RATE].iloc[0] == 0.0
