import numpy as np
import surfinBH

import logger
import schemas.binary


local_logger = logger.get_logger(__name__)


def load_fits(fits: schemas.binary.Fits) -> surfinBH.surfinBH.SurFinBH:
    """
    Load a surfinBH fits.

    Args:
    -----
        fits : schemas.binary.Fits
            The fits to load. The available fits are:
            - NRSur3dq8Remnant: non precessing BHs with mass ratio<=8, anti-/aligned spin <= 0.8
            - NRSur7dq4Remnant: precessing BHs with mass ratio<=4, generic spin <= 0.8
            - surfinBH7dq2: precessing BHs with mass ratio <= 2, generic spin <= 0.8

    Returns
    -----
        fits (surfinBH.surfinBH.SurFinBH):
            The loaded fits.
    """

    fits_name = fits.value
    local_logger.info(
        "Loading surfinBH %s, description: %s.",
        fits_name,
        surfinBH.fits_collection[fits_name].desc,
    )
    return surfinBH.LoadFits(fits_name)


def simulate_remnant(binary: schemas.binary.Binary, fits: surfinBH.surfinBH.SurFinBH) -> dict[str, float]:
    """
    Simulate the remnant of a binary.

    Args:
    -----
        binary (schemas.binary.Binary):
            The binary to simulate.

        fits (surfinBH.surfinBH.SurFinBH):
            The surrogate model.

    Returns:
    -----
        data (dict[str, float]):
            The simulated remnant parameters.
    """

    vf, _ = fits.vf(binary.mass_ratio, binary.chi1, binary.chi2)
    chif, _ = fits.chif(binary.mass_ratio, binary.chi1, binary.chi2)
    remnant_mass, _ = fits.mf(binary.mass_ratio, binary.chi1, binary.chi2)
    remannt_speed = np.sqrt(np.dot(vf, vf)) * schemas.binary.SPEED_OF_LIGHT
    remnant_spin = np.sqrt(np.dot(chif, chif))

    data = {
        "m1": binary.m1,
        "m2": binary.m2,
        "q": binary.mass_ratio,
        "mf": remnant_mass,
        "vf": remannt_speed,
        "chif": remnant_spin,
    }
    return data
