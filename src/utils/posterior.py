import h5py
import pandas as pd

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

    def __init__(self, df: pd.DataFrame, is_mass_injected: bool, sampling=10) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame
            The prior dataframe.
        is_mass_injected : bool
            Whether the mass is injected.
        sampling : int
            Number of samples to be drawn each time.
        """
        self.sampling = sampling
        self.is_mass_injected = is_mass_injected
        self.prior = df

        if self.is_mass_injected:
            self.prior["mf_"] = self.prior["mf"] * (self.prior["m1"] + self.prior["m2"])

        logger.info("Initialized posterior sampler.")
        logger.info("Is mass injected: %s", self.is_mass_injected)
        logger.info("Number of samples in prior: %d", len(self.prior))
        logger.info("Sampling amount each time: %d", self.sampling)

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
            samples = self.prior.loc[(self.prior["mf_"] - mass_measure).abs() < 0.5]
            if len(samples) < self.sampling:
                logger.warning("Not enough similar samples in the prior.")
            samples = samples.iloc[
                (samples["chif"] - spin_measure).abs().argsort()[: self.sampling]
            ]

        if not self.is_mass_injected:
            samples = self.prior.iloc[
                (self.prior["chif"] - spin_measure).abs().argsort()[: self.sampling]
            ]
            samples["m1"] = (
                mass_measure / samples["mf"] * samples["q"] / (1 + samples["q"])
            )
            samples["m2"] = mass_measure / samples["mf"] / (1 + samples["q"])
            samples["mf_"] = mass_measure

        return samples
