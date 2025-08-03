import os

import pytest
import surfinBH

from archeo.constants import Fits
from archeo.core.prior import Prior
from archeo.core.simulator import Simulator
from archeo.schema import Domain, PriorConfig


# Ensure the model fits is downloaded during tests
fits_filepath = os.path.dirname(surfinBH.__file__) + "/data/fit_3dq8.h5"
if os.path.exists(fits_filepath):
    os.remove(fits_filepath)


@pytest.fixture(name="uniform_mass_aligned_spin_prior_config")
def default_uniform_mass_aligned_spin_prior_config():
    """Create a default prior config for testing."""

    return PriorConfig(
        n_samples=1000,
        fits=Fits.NRSUR3DQ8REMNANT,
        is_mahapatra=False,
        is_spin_aligned=True,
        is_only_up_aligned_spin=False,
        is_uniform_in_mass_ratio=False,
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


@pytest.fixture(name="uniform_q_aligned_spin_spin_prior_config")
def default_uniform_q_aligned_spin_spin_prior_config():
    """Create a default prior config for testing."""

    return PriorConfig(
        n_samples=1000,
        fits=Fits.NRSUR3DQ8REMNANT,
        is_mahapatra=False,
        is_spin_aligned=True,
        is_only_up_aligned_spin=False,
        is_uniform_in_mass_ratio=True,
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


def test_run_simulation_1(uniform_mass_aligned_spin_prior_config):
    """Run a simulation with the default prior config."""

    prior = Prior.from_config(uniform_mass_aligned_spin_prior_config)

    assert len(prior) == 1000


def test_run_simulation_2(uniform_q_aligned_spin_spin_prior_config):
    """Run a simulation with the default prior config."""

    prior = Prior.from_config(uniform_q_aligned_spin_spin_prior_config)

    assert len(prior) == 1000


def test_run_2g1g_simulation(uniform_mass_aligned_spin_prior_config):
    """Run a simulation with the default prior config."""

    prior = Prior.from_config(uniform_mass_aligned_spin_prior_config)
    simulator = Simulator(uniform_mass_aligned_spin_prior_config)
    simulator.use_remnant_results(df_bh1=prior)

    prior_2g = Prior.from_simulator(simulator)
    assert len(prior_2g) == 1000


def test_run_2g2g_simulation(uniform_mass_aligned_spin_prior_config):
    """Run a simulation with the default prior config."""

    prior = Prior.from_config(uniform_mass_aligned_spin_prior_config)
    simulator = Simulator(uniform_mass_aligned_spin_prior_config)
    simulator.use_remnant_results(df_bh1=prior, df_bh2=prior)

    prior_2g = Prior.from_simulator(simulator)
    assert len(prior_2g) == 1000
