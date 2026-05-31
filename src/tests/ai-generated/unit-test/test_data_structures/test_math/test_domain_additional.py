import numpy as np

from archeo.data_structures.math import Domain, PiecewiseDomain


def test_domain_contains_and_not_contains_vectorized_inclusive_bounds():
    domain = Domain(low=1.0, high=2.0)
    x = np.array([0.5, 1.0, 1.5, 2.0, 2.5])

    contains = domain.contains(x)
    not_contains = domain.not_contains(x)

    assert np.array_equal(contains, np.array([False, True, True, True, False]))
    assert np.array_equal(not_contains, np.array([True, False, False, False, True]))


def test_domain_not_contains_scalar_is_complement_of_contains():
    domain = Domain(low=-3.0, high=7.0)

    for value in [-10.0, -3.0, 0.0, 7.0, 9.0]:
        assert domain.not_contains(value) is (not domain.contains(value))


def test_domain_to_tuple_returns_bounds_exactly():
    domain = Domain(low=1.25, high=9.75)
    assert domain.to_tuple() == (1.25, 9.75)


def test_domain_default_infinite_bounds_include_finite_values():
    domain = Domain()  # [-inf, inf]
    x = np.array([-1e9, -1.0, 0.0, 1.0, 1e9])

    assert np.array_equal(domain.contains(x), np.array([True, True, True, True, True]))
    assert np.array_equal(domain.not_contains(x), np.array([False, False, False, False, False]))


def test_piecewise_domain_vectorized_discrete_union():
    domain = PiecewiseDomain(domains=[Domain(low=1.0, high=2.0), Domain(low=4.0, high=5.0)])
    x = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 5.0, 6.0])

    contains = domain.contains(x)
    not_contains = domain.not_contains(x)

    assert np.array_equal(contains, np.array([False, True, True, True, False, True, True, False]))
    assert np.array_equal(not_contains, np.array([True, False, False, False, True, False, False, True]))


def test_piecewise_domain_vectorized_continuous_union_with_shared_boundary():
    domain = PiecewiseDomain(domains=[Domain(low=1.0, high=2.0), Domain(low=2.0, high=3.0)])
    x = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

    assert np.array_equal(domain.contains(x), np.array([False, True, True, True, True, True, False]))
    assert np.array_equal(domain.not_contains(x), np.array([True, False, False, False, False, False, True]))


def test_piecewise_domain_not_contains_matches_logical_negation_for_regular_values():
    domain = PiecewiseDomain(domains=[Domain(low=-2.0, high=-1.0), Domain(low=1.0, high=2.0)])
    x = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])

    contains = domain.contains(x)
    not_contains = domain.not_contains(x)

    assert np.array_equal(not_contains, np.logical_not(contains))


def test_piecewise_domain_empty_scalar_behavior():
    # Important edge behavior of current implementation:
    # any([]) -> False, all([]) -> True for scalar checks
    domain = PiecewiseDomain(domains=[])

    assert domain.contains(1.23) is False
    assert domain.not_contains(1.23) is True
