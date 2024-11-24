import os

import archeo
from archeo.constants import Columns as C


def test_self_estimate():
    """Run an estimation on the test dataset,
    where we let prior=posterior. Suppose we should get 100% recovery rate.
    """

    prior = archeo.Prior.from_feather(f"{os.path.dirname(os.path.dirname(__file__))}/test_data/prior.feather")
    mass_posterior = prior[C.BH_MASS]
    spin_posterior = prior[C.BH_SPIN]

    # Run the estimation
    posterior = prior.to_posterior(mass_posterior, spin_posterior)

    assert posterior[C.RECOVERY_RATE].iloc[0] == 1.0
