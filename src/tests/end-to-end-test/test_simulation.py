import pytest

from archeo.constants import Fits
from archeo.core.prior import Prior
from archeo.schema import Domain, PriorConfig


@pytest.fixture(name="prior_config")
def default_prior_config():
    """Create a default prior config for testing."""

    return PriorConfig(
        n_samples=1000,
        fits=Fits.NRSUR3DQ8REMNANT,
        is_mahapatra=False,
        is_spin_aligned=True,
        is_only_up_aligned_spin=False,
        a_1=Domain(low=0.0, high=1.0),
        a_2=Domain(low=0.0, high=1.0),
        phi_1=Domain(low=0.0, high=2.0),
        phi_2=Domain(low=0.0, high=2.0),
        theta_1=Domain(low=0.0, high=1.0),
        theta_2=Domain(low=0.0, high=1.0),
        m_1=Domain(low=5.0, high=65.0),
        m_2=Domain(low=5.0, high=65.0),
        mass_ratio=Domain(low=1.0, high=2.0),
    )


def test_run_simulation(prior_config):
    """Run a simulation with the default prior config."""

    prior = Prior.from_config(prior_config)

    assert len(prior) == 1000
