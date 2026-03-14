from archeo.data_structures.math import Domain


def test_domain():

    domain = Domain(low=1, high=2)

    assert not domain.contains(0.5)
    assert domain.contains(1.0)
    assert domain.contains(1.5)
    assert domain.contains(2.0)
    assert not domain.contains(2.5)
