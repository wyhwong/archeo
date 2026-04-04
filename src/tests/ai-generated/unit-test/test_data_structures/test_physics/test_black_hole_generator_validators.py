import numpy as np
import pytest

from archeo.data_structures.distribution import Uniform
from archeo.data_structures.physics.black_hole import BlackHoleGenerator


def test_black_hole_generator_rejects_invalid_spin_magnitude_distribution():
    with pytest.raises(ValueError, match="Spin magnitude distribution must be within the range \\[0, 1\\]"):
        BlackHoleGenerator(spin_magnitude_distribution=Uniform(low=-0.1, high=0.9))


def test_black_hole_generator_rejects_invalid_phi_distribution():
    with pytest.raises(ValueError, match="Phi distribution must be within the range \\[0, 2 \\* pi\\]"):
        BlackHoleGenerator(phi_distribution=Uniform(low=0.0, high=2 * np.pi + 0.01))


def test_black_hole_generator_rejects_invalid_theta_distribution():
    with pytest.raises(ValueError, match="Theta distribution must be within the range \\[0, pi\\]"):
        BlackHoleGenerator(theta_distribution=Uniform(low=-0.01, high=np.pi))
