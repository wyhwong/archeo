import surfinBH
import numpy as np

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
    newUnitVector : np.ndarray
        Cartesian coordinates.
    """
    newUnitVector = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    return np.array(newUnitVector, dtype=float)


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


class BinaryGenerator:
    """
    Binary generator.
    NOTE:
        Convention:
            1. Heavier black hole
            2. Lighter black hole
    """

    def __init__(self, config: schemas.binary.BinaryConfig) -> None:
        """
        Initialize the binary generator.

        Parameters
        ----------
        config : schemas.binary.BinaryConfig
            Configuration of the binary generator.

        Returns
        -------
        None
        """
        self.config = config

    def __call__(self) -> schemas.binary.Binary:
        """
        Generate a binary.

        Returns
        -------
        binary : schemas.binary.Binary
            Binary.
        """
        return schemas.binary.Binary(generate_parameter(self.config.mass_ratio), self._get_spin(), self._get_spin())

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
