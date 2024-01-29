import surfinBH
import numpy as np
import scipy.constants

import schemas.binary


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
    remannt_speed = np.sqrt(np.dot(vf, vf)) * scipy.constants.speed_of_light / 1000.0
    remnant_spin = np.sqrt(np.dot(chif, chif))

    data = {
        "m1": binary.m1,
        "m2": binary.m2,
        "mass_ratio": binary.mass_ratio,
        "mf": remnant_mass,
        "vf": remannt_speed,
        "chif": remnant_spin,
    }
    return data
