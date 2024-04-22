from typing import Callable

import numpy as np
import pandas as pd

import archeo.logger
import archeo.schemas.common


local_logger = archeo.logger.get_logger(__name__)


def is_in_bounds(value: float, domain: archeo.schemas.common.Domain) -> bool:
    """
    Check if the value is in the domain.

    Args:
    -----
        value (float):
            The value to check.

        domain (archeo.schemas.common.Domain):
            The domain to check the value against.

    Returns:
    -----
        is_in_bound (bool):
            True if the value is in the domain, False otherwise.
    """

    in_bound = True

    if domain.low and value < domain.low:
        in_bound = False

    if domain.high and value > domain.high:
        in_bound = False

    return in_bound


def get_generator_from_domain(domain: archeo.schemas.common.Domain) -> Callable:
    """
    Get a generator function from a domain.

    Args:
    -----
        domain (archeo.schemas.common.Domain):
            The domain to generate value from.

    Returns:
    -----
        generate_value_from_domain (Callable):
            Generator function.
    """

    def generate_value_from_domain() -> float:
        """
        Generate a value in the domain.

        Args:
        -----
            domain (archeo.schemas.common.Domain):
                The domain to generate value from.

        Returns:
        -----
            value (float):
                The generated value.
        """

        if domain.low is None or domain.high is None:
            local_logger.error("Both low and high must be specified.")
            raise ValueError("Both low and high must be specified.")

        return np.random.uniform(low=domain.low, high=domain.high)

    return generate_value_from_domain


def sph2cart(theta: float, phi: float) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Args:
    -----
        theta (float):
            Polar angle.

        phi (float):
            Azimuthal angle.

    Returns:
    -----
        cartesian_vector (np.ndarray):
            Cartesian coordinates.
    """

    cartesian_vector = [
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ]
    return np.array(cartesian_vector, dtype=float)


def get_generator_from_csv(csv_path: str) -> Callable:
    """
    Get a generator function from a csv file.

    Args:
    -----
        csv_path (str):
            Path to the csv file.

    Returns:
    -----
        parameter_from_pdf (Callable):
            Generator function.
    """

    local_logger.info("Reading PDF from %s...", csv_path)
    df = pd.read_csv(csv_path)
    x, p = df["x"].values, df["y"].values

    def generate_value_from_pdf() -> float:
        """
        Generate a value for a parameter.

        Returns:
        -----
            value (float):
                The generated value.
        """

        return np.random.choice(x, p=p)

    return generate_value_from_pdf
