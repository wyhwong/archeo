import pandas as pd
from typing import Optional

import core.utils
import logger


local_logger = logger.get_logger(__name__)


def get_posterior_from_json(filepath: str) -> dict:
    """
    Get the posterior from a json file.

    Args:
    -----
        filepath (str):
            Filepath of the json file.

    Returns:
    -----
        posterior (dict):
            Content of the posterior.
    """

    return core.utils.load_json(filepath=filepath)["posterior"]["content"]


def read_posterior_from_h5(filepath: str, fits: str = "NRSur7dq4") -> dict:
    """
    Read posterior from h5 file.

    Args:
    -----
        filepath (str):
            Filepath of the h5 file.

        fits (str):
            Fits to read from.

    Returns:
    -----
        posterior (dict):
            Content of the posterior.
    """

    return core.utils.load_h5(filepath)[fits]["posterior_samples"]


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
        Initialize the posterior sampler.

        Args:
        -----
            df (pd.DataFrame):
                The prior dataframe.

            is_mass_injected (bool):
                Whether the mass is injected.

            n_sample (float):
                The number of samples to be sampled each time.

            spin_tolerance (float):
                The tolerance of the spin.

            mass_tolerance (Optional[float]):
                The tolerance of the mass.

        Returns:
        -----
            None
        """

        self._prior = df
        self.is_mass_injected = is_mass_injected
        self.n_sample = n_sample
        self.spin_tolerance = spin_tolerance
        self.mass_tolerance = mass_tolerance

        if self.is_mass_injected:
            if not mass_tolerance:
                raise ValueError("Mass tolerance must be specified if mass is injected.")
            self._prior["mf_"] = self._prior["mf"] * (self._prior["m1"] + self._prior["m2"])

        local_logger.info(
            "Constructed a posterior sampler: mass injected: %s, spin tol: %.2f, mass tol: %.2f config: %s",
            self.is_mass_injected,
            self.spin_tolerance,
            self.mass_tolerance,
            self.config,
        )

    def infer_parental_params(self, spin_measure: float, mass_measure: float) -> pd.DataFrame:
        """
        Infer the parental parameters from the posterior.

        Args:
        -----
            spin_measure (float):
                The spin measure.

            mass_measure (float):
                The mass measure.

        Returns:
        -----
            samples (pd.DataFrame):
                The sampled parental parameters.
        """

        if self.is_mass_injected:
            df = self._prior.loc[
                ((self._prior["mf_"] - mass_measure).abs() < self.mass_tolerance)
                & ((self._prior["chif"] - spin_measure).abs() < self.spin_tolerance)
            ]
            samples = self.sample_from_df(df)
        else:
            df = self._prior.loc[((self._prior["chif"] - spin_measure).abs() < self.spin_tolerance)]
            samples = self.sample_from_df(df)
            samples["m1"] = mass_measure / samples["mf"] * samples["q"] / (1 + samples["q"])
            samples["m2"] = mass_measure / samples["mf"] / (1 + samples["q"])
            samples["mf_"] = mass_measure

        return samples

    def sample_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sample from a dataframe.

        Args:
        -----
            df (pd.DataFrame):
                The dataframe to sample from.

        Returns:
        -----
            df (pd.DataFrame):
                The sampled dataframe.
        """

        if df.empty:
            local_logger.warning("No similar samples in the prior.")
        elif len(df) < self.n_sample:
            local_logger.warning("Not enough similar samples in the prior.")
        else:
            df = df.sample(self.n_sample)
        return df
