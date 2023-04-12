import numpy as np
import surfinBH
from numpy.random import uniform, rand
from scipy import constants

from .logger import get_logger

LOGGER = get_logger(logger_name="Utils | Merger")


def sph2cart(theta: float, phi: float) -> np.ndarray:
    newUnitVector = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    LOGGER.debug(
        f"Converted unit vector in spherical coordinate to cartesian coordinate: [1, {theta}, {phi}] -> {newUnitVector}"
    )
    return np.array(newUnitVector, dtype=float)


def generate_parameter(para_domain: dict) -> float:
    return uniform(low=para_domain["min"], high=para_domain["max"])


def load_fits(fits_name: str):
    LOGGER.info(f"Loading surfinBH {fits_name=}, description: {surfinBH.fits_collection[fits_name].desc}.")
    return surfinBH.LoadFits(fits_name)


class Binary:
    def __init__(self, fits, mass_ratio: float, chi1: np.ndarray, chi2: np.ndarray) -> None:
        LOGGER.debug(f"Initializing merger with {mass_ratio=}, {chi1=}, {chi2}")
        self.fits = fits
        self.mass_ratio = mass_ratio
        self.chi1, self.chi2 = chi1, chi2
        LOGGER.debug(f"Initialized merger.")

    # Output results: list
    # merger_params: [mass_ratio, spin_1, spin_2, remnant_mass, error, remnant_spin, error, remnant_speed, error]
    def merge(self) -> list:
        merger_params = [self.mass_ratio, np.dot(self.chi1, self.chi1), np.dot(self.chi2, self.chi2)]
        LOGGER.debug("Generating the remnant parameters for a merger.")
        for para in ["mf", "chif", "vf"]:
            remnant_para, err = getattr(self.fits, para)(self.mass_ratio, self.chi1, self.chi2)
            if para == "mf":
                LOGGER.debug(f"Computed remaining mass percentage: {remnant_para}%")
            elif para == "chif":
                LOGGER.debug(f"Computed remnant spin: {remnant_para}")
                remnant_para = np.sqrt(np.dot(remnant_para, remnant_para))
                err = np.sqrt(np.dot(err, err))
            elif para == "vf":
                LOGGER.debug(f"Computed remnant speed: {remnant_para} km/s")
                remnant_para = np.sqrt(np.dot(remnant_para, remnant_para)) * constants.speed_of_light / 1000.0
                err = np.sqrt(np.dot(err, err)) * constants.speed_of_light / 1000.0
            merger_params += [remnant_para, err]
        LOGGER.debug(f"Generated the remnant parameters for a merger: {merger_params}")
        return merger_params


class BinaryParamsGenerator:
    def __init__(self, config: dict) -> None:
        LOGGER.debug(f"Initializing a random binary generator from config: {config}...")
        self.config = config
        self.config["phi"]["min"] *= np.pi
        self.config["phi"]["max"] *= np.pi
        self.config["theta"]["min"] *= np.pi
        self.config["theta"]["max"] *= np.pi
        LOGGER.debug("Initialized random binary generator.")

    def __call__(self) -> tuple:
        # Convention:
        #   1. Heavier black hole
        #   2. Lighter black hole
        if self.config["massInjection"]:
            mass_ratio = 0
            while not (self.config["massRatio"]["min"] <= mass_ratio <= self.config["massRatio"]["max"]):
                mass1, mass2 = generate_parameter(self.config["mass"]), generate_parameter(self.config["mass"])
                mass1, mass2 = max(mass1, mass2), min(mass1, mass2)
                mass_ratio = mass1 / mass2
        else:
            mass1, mass2 = None, None
            mass_ratio = generate_parameter(self.config["massRatio"])
        chi1, chi2 = self.get_spin()
        LOGGER.debug(f"Output binary setting: {mass_ratio=}, {chi1=}, {chi2=}.")
        return (mass_ratio, chi1, chi2, mass1, mass2)

    def get_spin(self) -> list:
        spins = []
        for _ in range(2):
            spin = generate_parameter(self.config["spin"])
            if self.config["align"]:
                phi = 0.0
                theta = 0.0 + np.round(rand()) * np.pi
            else:
                phi = generate_parameter(self.config["phi"])
                theta = generate_parameter(self.config["theta"])
            univ = sph2cart(theta, phi)
            spins.append(spin * univ)
        return spins
