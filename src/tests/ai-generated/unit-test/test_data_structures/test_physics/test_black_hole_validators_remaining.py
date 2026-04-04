import numpy as np
import pytest

from archeo.data_structures.distribution import Uniform
from archeo.data_structures.physics.black_hole import BlackHoleGenerator


def test_spin_validator_raise_line():
    with pytest.raises(ValueError):
        BlackHoleGenerator(spin_magnitude_distribution=Uniform(low=-1.0, high=0.5))


def test_phi_validator_raise_line():
    with pytest.raises(ValueError):
        BlackHoleGenerator(phi_distribution=Uniform(low=0.0, high=2 * np.pi + 1e-6))


def test_theta_validator_raise_line():
    with pytest.raises(ValueError):
        BlackHoleGenerator(theta_distribution=Uniform(low=-1e-6, high=np.pi))
