import numpy as np

import archeo.logger


local_logger = archeo.logger.get_logger(__name__)


def sph2cart(theta: float, phi: float) -> np.ndarray:
    """Convert spherical coordinates to Cartesian coordinates.

    Args:
        theta (float): Polar angle.
        phi (float): Azimuthal angle.

    Returns:
        cartesian_vector (np.ndarray): Cartesian coordinates.
    """

    cartesian_vector = [
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ]
    return np.array(cartesian_vector, dtype=float)
