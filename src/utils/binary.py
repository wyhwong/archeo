import surfinBH
import numpy as np
import pandas as pd
from typing import Callable

import utils.common
import utils.logger
import schemas.common
import schemas.binary

logger = utils.logger.get_logger(logger_name="utils|fits")


def load_fits(fits_name: str) -> surfinBH.surfinBH.SurFinBH:
    """
    Load a fits file.

    Parameters
    ----------
    fits_name : str
        Name of the fits file.

    Returns
    -------
    fits : surfinBH.surfinBH.SurFinBH
        Gravitational wave waveform.
    """
    logger.info(f"Loading surfinBH {fits_name=}, description: {surfinBH.fits_collection[fits_name].desc}.")
    return surfinBH.LoadFits(fits_name)


def sph2cart(theta: float, phi: float) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    theta : float
        Polar angle.
    phi : float
        Azimuthal angle.

    Returns
    -------
    new_unit_vector : np.ndarray
        Cartesian coordinates.
    """
    new_unit_vector = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    return np.array(new_unit_vector, dtype=float)


def generate_parameter(domain: schemas.common.Domain) -> float:
    """
    Generate a parameter.

    Parameters
    ----------
    domain : schemas.common.Domain
        Domain of the parameter.

    Returns
    -------
    parameter : float
        Generated parameter.
    """
    return np.random.uniform(low=domain.low, high=domain.high)


def get_generator_from_csv(csv_path: str) -> Callable:
    """
    Generate a generator function from a csv file.

    Parameters
    ----------
    csv_path : str
        Path to the csv file.

    Returns
    -------
    generator_func : Callable
        Generator function to generate values for a paramter.
    """
    df = pd.read_csv(csv_path)
    x, p = df["x"].values, df["y"].values

    def parameter_from_pdf() -> float:
        """
        Generate a value for a parameter.

        Returns
        -------
        parameter : float
            Value of the parameter.
        """
        return np.random.choice(x, p=p)

    return parameter_from_pdf


class BinaryGenerator:
    """
    Binary generator.
    NOTE:
        Convention:
            1. Heavier black hole
            2. Lighter black hole
    """

    def __init__(
        self,
        config: schemas.binary.BinaryConfig,
        mass_injection: bool = False,
        mass_ratio_from_pdf: Callable | None = None,
        mass_from_pdf: Callable | None = None,
    ) -> None:
        """
        Initialize the binary generator.

        Parameters
        ----------
        config : schemas.binary.BinaryConfig
            Configuration of the binary generator.
        mass_injection : bool, optional
            Whether to inject the masses, by default False
        mass_ratio_from_pdf : Callable, optional
            Generate mass ratio from pdf, by default None
        mass_from_pdf : Callable, optional
            Generate mass from pdf, by default None

        Returns
        -------
        None
        """
        self.config = config
        self.mass_injection = mass_injection
        self.mass_ratio_from_pdf = mass_ratio_from_pdf
        self.mass_from_pdf = mass_from_pdf

        if self.mass_from_pdf and self.mass_ratio_from_pdf:
            raise ValueError("Both mass_from_pdf and mass_ratio_from_pdf exist.")

    def __call__(self) -> schemas.binary.Binary:
        """
        Generate a binary.

        Returns
        -------
        binary : schemas.binary.Binary
            Binary.
        """
        chi1, chi2 = self._get_spin(), self._get_spin()

        if self.mass_injection:
            m1, m2 = self._get_masses()
            mass_ratio = m1 / m2

        else:
            m1, m2 = None

            if self.mass_ratio_from_pdf:
                mass_ratio = self.mass_ratio_from_pdf()
            else:
                mass_ratio = generate_parameter(self.config.mass_ratio)

        return schemas.binary.Binary(mass_ratio, chi1, chi2, m1, m2)

    def simulate(self, num: int) -> list[schemas.binary.Binary]:
        """
        Simulate binaries.

        Parameters
        ----------

        Returns
        -------
        binaries : list[schemas.binary.Binary]
            Binaries.
        """
        return [self.__call__() for _ in range(num)]

    def _get_spin(self) -> np.ndarray:
        """
        Generate the spin of the binary.

        Returns
        -------
        spin : np.ndarray
            Spin of a black hole in the binary.
        """
        spin = generate_parameter(self.config.spin)
        if self.config.aligned_spin:
            phi = 0.0
            theta = 0.0 + np.round(np.random.rand()) * np.pi
        else:
            phi = generate_parameter(self.config.phi)
            theta = generate_parameter(self.config.theta)

        univ = sph2cart(theta, phi)
        return spin * univ

    def _get_masses(self) -> tuple[float, float]:
        """
        Generate the masses of the binary.

        Returns
        -------
        m1 : float
            Mass of the heavier black hole.
        m2 : float
            Mass of the lighter black hole.
        """
        in_bound = False
        while not in_bound:
            if self.mass_from_pdf:
                m_1, m_2 = self.mass_from_pdf(), self.mass_from_pdf()
            else:
                m_1, m_2 = generate_parameter(self.config.mass), generate_parameter(self.config.mass)

            m1, m2 = max(m_1, m_2), min(m_1, m_2)
            if self.config.mass_ratio.in_bound(m1 / m2):
                in_bound = True

        return (m1, m2)
