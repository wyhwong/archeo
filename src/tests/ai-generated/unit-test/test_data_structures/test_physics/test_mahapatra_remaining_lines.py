from archeo.data_structures.math import Domain
from archeo.data_structures.physics.mahapatra import MahapatraMassFunction


def test_mahapatra_min_max_and_scalar_draw():
    d = Domain(low=5.0, high=9.0)
    mf = MahapatraMassFunction(mass=d, dm=1.0, resolution=0.2)
    tol = 1e-7

    assert mf.min == 5.0
    assert mf.max == 9.0

    x = mf.draw()  # scalar path
    assert mf.min - tol <= x <= mf.max + tol
