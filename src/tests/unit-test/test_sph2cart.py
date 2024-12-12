import numpy as np

from archeo.utils.math import sph2cart


def test_positive_z_conversion():
    """Test the conversion from spherical to Cartesian coordinates."""

    x, y, z = sph2cart(theta=0, phi=0)
    assert np.isclose(x, 0, atol=1e-10)
    assert np.isclose(y, 0, atol=1e-10)
    assert np.isclose(z, 1, atol=1e-10)


def test_negative_z_conversion():
    """Test the conversion from spherical to Cartesian coordinates."""

    x, y, z = sph2cart(theta=np.pi, phi=0)
    assert np.isclose(x, 0, atol=1e-10)
    assert np.isclose(y, 0, atol=1e-10)
    assert np.isclose(z, -1, atol=1e-10)


def test_positive_y_conversion():
    """Test the conversion from spherical to Cartesian coordinates."""

    x, y, z = sph2cart(theta=np.pi / 2, phi=np.pi / 2)
    assert np.isclose(x, 0, atol=1e-10)
    assert np.isclose(y, 1, atol=1e-10)
    assert np.isclose(z, 0, atol=1e-10)


def test_negative_y_conversion():
    """Test the conversion from spherical to Cartesian coordinates."""

    x, y, z = sph2cart(theta=np.pi / 2, phi=3 * np.pi / 2)
    assert np.isclose(x, 0, atol=1e-10)
    assert np.isclose(y, -1, atol=1e-10)
    assert np.isclose(z, 0, atol=1e-10)


def test_positive_x_conversion():
    """Test the conversion from spherical to Cartesian coordinates."""

    x, y, z = sph2cart(theta=np.pi / 2, phi=0)
    assert np.isclose(x, 1, atol=1e-10)
    assert np.isclose(y, 0, atol=1e-10)
    assert np.isclose(z, 0, atol=1e-10)


def test_negative_x_conversion():
    """Test the conversion from spherical to Cartesian coordinates."""

    x, y, z = sph2cart(theta=np.pi / 2, phi=np.pi)
    assert np.isclose(x, -1, atol=1e-10)
    assert np.isclose(y, 0, atol=1e-10)
    assert np.isclose(z, 0, atol=1e-10)
