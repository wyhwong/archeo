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

    Parameters
    ----------
    config : dict
        Configuration of the binary generator.
    """

    def __init__(self, config: schemas.binary.BinaryConfig) -> None:
        self.config = config

    def __call__(self) -> tuple:
        # Convention:
        #   1. Heavier black hole
        #   2. Lighter black hole
        mass_ratio = generate_parameter(self.config["massRatio"])
        chi1, chi2 = self.get_spin()
        return (mass_ratio, chi1, chi2)

    def get_spin(self) -> list:
        spins = []
        for _ in range(2):
            spin = generate_parameter(self.config["spin"])
            if self.config["align"]:
                phi = 0.0
                theta = 0.0 + np.round(np.random.rand()) * np.pi
            else:
                phi = generate_parameter(self.config["phi"])
                theta = generate_parameter(self.config["theta"])
            univ = sph2cart(theta, phi)
            spins.append(spin * univ)
        return spins
