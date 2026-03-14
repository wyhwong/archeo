from archeo.data_structures.math import Domain, PiecewiseDomain


def test_discreate_piecewise_domain():

    domain = PiecewiseDomain(domains=[Domain(low=1, high=2), Domain(low=3, high=4)])

    assert not domain.contains(0.5)
    assert domain.contains(1.0)
    assert domain.contains(1.5)
    assert domain.contains(2.0)
    assert not domain.contains(2.5)
    assert domain.contains(3.0)
    assert domain.contains(3.5)
    assert domain.contains(4.0)
    assert not domain.contains(4.5)


def test_continuous_piecewise_domain():

    domain = PiecewiseDomain(domains=[Domain(low=1, high=2), Domain(low=2, high=3)])

    assert not domain.contains(0.5)
    assert domain.contains(1.0)
    assert domain.contains(1.5)
    assert domain.contains(2.0)
    assert domain.contains(2.5)
    assert domain.contains(3.0)
    assert not domain.contains(3.5)
