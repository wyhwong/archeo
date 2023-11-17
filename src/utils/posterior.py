import h5py
import pandas as pd
from typing import Optional

import utils.common
import utils.logger


logger = utils.logger.get_logger(logger_name="utils|posterior")


def read_posterior_from_json(filepath: str) -> dict:
    """
    Read the posterior of parameter estimation from a json file.

    Parameters
    ----------
    filepath : str
        Path to the json file.

    Returns
    -------
    posterior : dict
        Dictionary containing the posterior.
    """
    logger.info("Reading posterior from %s...", filepath)
    return utils.common.read_dict_from_json(filepath=filepath)["posterior"]["content"]


def read_posterior_from_h5(filepath: str, fits="NRSur7dq4") -> dict:
    """
    Read the posterior of parameter estimation from a h5 file.

    Parameters
    ----------
    filepath : str
        Path to the h5 file.

    fits : str
        Name of the waveform model.
    """
    logger.info("Reading posterior from %s...", filepath)
    return h5py.File(filepath, "r")[fits]["posterior_samples"]


class PosteriorSampler:
    """
    Posterior sampler for parameter estimation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        is_mass_injected: bool,
        n_sample: float = 10,
        spin_tolerance: float = 0.05,
        mass_tolerance: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame
            The prior dataframe.
        is_mass_injected : bool
            Whether the mass is injected.
        n_sample : int
            Number of samples to be drawn from the prior.
        spin_tolerance : float
            Tolerance of the spin measurement.
        mass_tolerance : float
            Tolerance of the mass measurement.
        """
        self.prior = df
        self.is_mass_injected = is_mass_injected
        self.n_sample = n_sample
        self.spin_tolerance = spin_tolerance
        self.mass_tolerance = mass_tolerance

        if self.is_mass_injected:
            if not mass_tolerance:
                raise ValueError(
                    "Mass tolerance must be specified if mass is injected."
                )
            self.prior["mf_"] = self.prior["mf"] * (self.prior["m1"] + self.prior["m2"])

        logger.info("Initialized posterior sampler.")
        logger.info("Is mass injected: %s", self.is_mass_injected)
        logger.info("Number of samples in prior: %d", len(self.prior))
        logger.info("Sampling amount each time: %d", self.n_sample)
        logger.info("Spin tolerance: %f", self.spin_tolerance)
        logger.info("Mass tolerance: %f", self.mass_tolerance)

    def infer_parental_params(
        self, spin_measure: float, mass_measure: float
    ) -> pd.DataFrame:
        """
        Infer the parental parameters from the posterior of the child parameters.

        Parameters
        ----------
        spin_measure : float
            The spin measurement of the remnant
        mass_measure : float
            The mass measurement of the remnant

        Returns
        -------
        samples : pd.DataFrame
            The samples of the parental mass
        """

        if self.is_mass_injected:
            samples = self.prior.loc[
                ((self.prior["mf_"] - mass_measure).abs() < self.mass_tolerance)
                & ((self.prior["chif"] - spin_measure).abs() < self.spin_tolerance)
            ]

        else:
            samples = self.prior.loc[
                ((self.prior["chif"] - spin_measure).abs() < self.spin_tolerance)
            ]
            samples["m1"] = (
                mass_measure / samples["mf"] * samples["q"] / (1 + samples["q"])
            )
            samples["m2"] = mass_measure / samples["mf"] / (1 + samples["q"])
            samples["mf_"] = mass_measure

        if samples.empty:
            logger.warning("No similar samples in the prior.")
        elif len(samples) < self.n_sample:
            logger.warning("Not enough similar samples in the prior.")
        else:
            samples.sample(self.n_sample)

        return samples
