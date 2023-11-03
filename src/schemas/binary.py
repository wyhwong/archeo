import surfinBH
import numpy as np
import scipy.constants

import schemas.common


class Binary:
    """
    Binary class.

    Parameters
    ----------
    fits : surfinBH.surfinBH.SurFinBH
        Gravitational wave waveform
    mass_ratio : float
        Mass ratio of the binary
    chi1 : np.ndarray
        Spin of the heavier black hole
    chi2 : np.ndarray
        Spin of the lighter black hole
    """

    fits: surfinBH.surfinBH.SurFinBH
    mass_ratio: float
    chi1: np.ndarray
    chi2: np.ndarray

    # merger_params: [mass_ratio, spin_1, spin_2, remnant_mass, error, remnant_spin, error, remnant_speed, error]
    def __post_init__(self):
        """
        Post initialization.

        Attributes
        ----------
        vf : np.ndarray
            Final velocity of the remnant black hole
        err_vf : np.ndarray
            Error of the final velocity of the remnant black hole
        chif : np.ndarray
            Final spin of the remnant black hole
        err_chif : np.ndarray
            Error of the final spin of the remnant black hole
        mf : np.ndarray
            Final mass of the remnant black hole, [0, 1]
        err_mf : np.ndarray
            Error of the final mass of the remnant black hole
        final_speed : float
            Final speed of the remnant black hole (in km/s)
        final_spin : float
            Final spin of the remnant black hole, [0, 1]
        """
        self.vf, self.err_vf = self.fits.vf(self.mass_ratio, self.chi1, self.chi2)
        self.chif, self.err_chif = self.fits.chif(self.mass_ratio, self.chi1, self.chi2)
        self.mf, self.err_mf = self.fits.mf(self.mass_ratio, self.chi1, self.chi2)
        self.final_speed = np.sqrt(np.dot(self.vf, self.vf)) * scipy.constants.speed_of_light / 1000.0
        self.final_spin = np.sqrt(np.dot(self.chif, self.chif))


class BinaryConfig:
    """
    Binary configuration.

    Parameters
    ----------
    mass_ratio : schemas.common.Domain
        Domain of the mass ratio of the binary
    aligned_spin : bool
        Whether the spin is aligned
    spin : schemas.common.Domain
        Domain of the spin of the black holes
    phi : schemas.common.Domain
        Domain of the azimuthal angle
    theta : schemas.common.Domain
        Domain of the polar angle
    """

    aligned_spin: bool
    mass_ratio: schemas.common.Domain
    spin: schemas.common.Domain
    phi: schemas.common.Domain
    theta: schemas.common.Domain

    def from_dict(args: dict):
        """
        Convert a dictionary to a BinaryConfig.

        Parameters
        ----------
        args : dict
            Dictionary of the binary configuration
        """
        return BinaryConfig(
            aligned_spin=args["alignedSpin"],
            mass_ratio=schemas.common.Domain.from_dict(args["massRatio"]["low"], args["massRatio"]["high"]),
            spin=schemas.common.Domain.from_dict(args["spin"]["low"], args["spin"]["high"]),
            phi=schemas.common.Domain.from_dict(args["phi"]["low"] * np.pi, args["phi"]["high"] * np.pi),
            theta=schemas.common.Domain.from_dict(args["theta"]["low"] * np.pi, args["theta"]["high"] * np.pi),
        )
