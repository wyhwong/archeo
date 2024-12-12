import numpy as np

from archeo.core.mahapatra import get_mahapatra_mass_fn
from archeo.schema import Domain


def test_mahapatra_mass_fn():
    """Test the Mahapatra mass function."""

    mass_fn = get_mahapatra_mass_fn(mass=Domain(5.0, 65.0), n_samples=100)

    masses = np.array([mass_fn() for _ in range(1000)])

    assert np.all(masses >= 5.0)
    assert np.all(masses <= 65.0)
    assert 1 < len(np.unique(masses)) <= 100
