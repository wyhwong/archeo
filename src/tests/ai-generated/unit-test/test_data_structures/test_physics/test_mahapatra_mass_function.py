import numpy as np

from archeo.data_structures.math import Domain
from archeo.data_structures.physics.mahapatra import MahapatraMassFunction


def test_mahapatra_mass_function_properties_and_draw_are_valid():
    mf = MahapatraMassFunction(mass=Domain(low=5.0, high=10.0), alpha=2.3, dm=1.0, resolution=0.1)

    masses = mf.masses
    probis = mf.probis

    assert np.isclose(masses[0], 5.0)
    assert masses[-1] <= 10.0
    assert np.isclose(probis.sum(), 1.0)
    assert (probis >= 0).all()

    samples = mf.draw(size=1000)
    assert len(samples) == 1000
    assert (samples >= mf.min).all()
    assert (samples <= mf.max).all()


def test_mahapatra_smoothing_function_hits_both_piecewise_regions():
    mf = MahapatraMassFunction(mass=Domain(low=5.0, high=8.0), alpha=2.0, dm=1.0, resolution=0.5)
    xs = np.array([5.2, 5.7, 6.2, 7.5])  # values below/above low+dm = 6.0
    out = mf._smoothing_func(xs)  # pylint: disable=protected-access

    assert out.shape == xs.shape
    assert np.isfinite(out).all()
    assert (out > 0).all()
