import surfinBH
import numpy as np
import scipy.constants
import dataclasses

import schemas.common


@dataclasses.dataclass
class Binary:
    """
    Binary class.

    Parameters
    ----------
    mass_ratio : float
        Mass ratio of the binary
    chi1 : np.ndarray
        Spin of the heavier black hole
    chi2 : np.ndarray
        Spin of the lighter black hole
    """

    mass_ratio: float
    chi1: np.ndarray
    chi2: np.ndarray

    # merger_params: [mass_ratio, spin_1, spin_2, remnant_mass, error, remnant_spin, error, remnant_speed, error]
    def __post_init__(self):
        """
        Post initialization.

        Attributes
        ----------
        mf : float
            Final mass of the remnant black hole, [0, 1]
        vf : float
            Final speed of the remnant black hole (in km/s)
        chif : float
            Final spin of the remnant black hole, [0, 1]
        """
        self.mf = 0.0
        self.vf = 0.0
        self.chif = 0.0

    def get_remnant_params(self, fits: surfinBH.surfinBH.SurFinBH) -> list[float]:
        """
        Get the parameters of the binary remnant.

        Parameters
        ----------
        fits : surfinBH.surfinBH.SurFinBH
            Gravitational wave waveform

        Returns
        -------
        remnant_params : list[float]
            Parameters of the binary remnant.
        """
        vf, _ = fits.vf(self.mass_ratio, self.chi1, self.chi2)
        chif, _ = fits.chif(self.mass_ratio, self.chi1, self.chi2)
        self.mf, _ = fits.mf(self.mass_ratio, self.chi1, self.chi2)
        self.vf = np.sqrt(np.dot(vf, vf)) * scipy.constants.speed_of_light / 1000.0
        self.chif = np.sqrt(np.dot(chif, chif))
        return [self.mass_ratio, self.mf, self.vf, self.chif]


@dataclasses.dataclass
class BinaryConfig:
    """
    Binary configuration.

    Parameters
    ----------
    aligned_spin : bool
        Whether the spin is aligned
    mass_ratio : schemas.common.Domain
        Domain of the mass ratio of the binary
    spin : schemas.common.Domain
        Domain of the spin of the black holes
    phi : schemas.common.Domain
        Domain of the azimuthal angle
    theta : schemas.common.Domain
        Domain of the polar angle
    fits : surfinBH.surfinBH.SurFinBH
        Gravitational wave waveform
    """

    aligned_spin: bool
    mass_ratio: schemas.common.Domain
    spin: schemas.common.Domain
    phi: schemas.common.Domain
    theta: schemas.common.Domain
    fits: surfinBH.surfinBH.SurFinBH

    def from_dict(args: dict):
        """
        Convert a dictionary to a BinaryConfig.

        Parameters
        ----------
        args : dict
            Dictionary of the binary configuration
        """
        return BinaryConfig(
            aligned_spin=args["aligned_spin"],
            mass_ratio=schemas.common.Domain(args["mass_ratio"]["low"], args["mass_ratio"]["high"]),
            spin=schemas.common.Domain(args["spin"]["low"], args["spin"]["high"]),
            phi=schemas.common.Domain(args["phi"]["low"] * np.pi, args["phi"]["high"] * np.pi),
            theta=schemas.common.Domain(args["theta"]["low"] * np.pi, args["theta"]["high"] * np.pi),
            fits=surfinBH.LoadFits(args["fits"]),
        )
