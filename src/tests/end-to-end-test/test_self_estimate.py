import os

import numpy as np
import pytest

import archeo
from archeo.constants import Columns as C
from archeo.constants import Suffixes as S


@pytest.fixture(name="prior")
def default_prior():
    """Load the default prior for testing."""

    filepath = f"{os.path.dirname(os.path.dirname(__file__))}/test_data/prior.json"

    return archeo.Prior.from_json(filepath)


def test_self_estimate(prior):
    """Run an estimation on the test dataset,
    where we let prior=posterior. Suppose we should get 100% recovery rate.
    """

    mass_posterior = prior[S.FINAL(C.MASS)].copy()
    spin_posterior = prior[S.FINAL(C.SPIN_MAG)].copy()

    # Run the estimation on itself
    posterior = prior.to_posterior(mass_posterior, spin_posterior)
    assert posterior[C.RECOVERY_RATE].iloc[0] == 1.0
    assert np.isclose(posterior[C.KS_PV_FOR_MASS].iloc[0], 1.0, atol=1e-7)
    assert posterior[C.KS_TEST_FOR_MASS].iloc[0] < 0.03
    assert np.isclose(posterior[C.KS_PV_FOR_SPIN].iloc[0], 1.0, atol=1e-7)
    assert posterior[C.KS_TEST_FOR_SPIN].iloc[0] < 0.03
    assert posterior[C.SAMPLE_ID].notna().all()

    # Rescale the masses to have m_min' > m_max + mass tolerance
    # And rerun the estimation, suppose recovery rate is 0
    mass_posterior += mass_posterior.max() + prior._mass_tolerance  # pylint: disable=protected-access
    posterior = prior.to_posterior(mass_posterior, spin_posterior)
    assert posterior[C.RECOVERY_RATE].iloc[0] == 0.0
    assert posterior[C.KS_PV_FOR_MASS].isna().all()
    assert posterior[C.KS_TEST_FOR_MASS].isna().all()
    assert posterior[C.KS_PV_FOR_SPIN].isna().all()
    assert posterior[C.KS_TEST_FOR_SPIN].isna().all()
    assert posterior[C.SAMPLE_ID].isna().all()
