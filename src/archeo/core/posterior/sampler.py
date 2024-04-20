from typing import Any, Optional

import pandas as pd

import archeo.core.utils
import archeo.logger


local_logger = archeo.logger.get_logger(__name__)


def get_posterior_from_json(filepath: str) -> dict[str, Any]:
    """
    Get the posterior from a json file.
    NOTE: This format reading is specific to the data from Romero-Shaw et. al. (2020)

    Args:
    -----
        filepath (str):
            Filepath of the json file.

    Returns:
    -----
        posterior (dict):
            Content of the posterior.
    """

    return archeo.core.utils.load_json(filepath=filepath)["posterior"]["content"]


def get_posterior_from_h5(filepath: str, fits: str = "NRSur7dq4") -> dict[str, Any]:
    """
    Read posterior from h5 file.
    NOTE: This format reading is specific to the data from LDG.

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

    return archeo.core.utils.load_h5(filepath)[fits]["posterior_samples"]


class PosteriorSampler:
    """Posterior sampler for parameter estimation."""

    def __init__(
        self,
        df: pd.DataFrame,
        is_mass_injected: bool,
        n_sample: int,
        spin_tolerance: float,
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

            n_sample (int):
                The number of samples to be sampled each time.

            spin_tolerance (float):
                The tolerance of the spin.

            mass_tolerance (Optional[float]):
                The tolerance of the mass.
                NOTE: This is only required if mass is injected

        Returns:
        -----
            None
        """

        self._prior = df
        self._is_mass_injected = is_mass_injected
        self._n_sample = n_sample
        self._spin_tolerance = spin_tolerance
        self._mass_tolerance = mass_tolerance

        if self._is_mass_injected:
            if not mass_tolerance:
                raise ValueError("Mass tolerance must be specified if mass is injected.")
            self._prior["mf_"] = self._prior["mf"] * (self._prior["m1"] + self._prior["m2"])

        local_logger.info(
            "Constructed a posterior sampler [n=%d]: mass injected: %s, spin tol: %.2f, mass tol: %.2f.",
            self._n_sample,
            self._is_mass_injected,
            self._spin_tolerance,
            self._mass_tolerance,
        )
        local_logger.info("Prior summary: %s.", self._prior.describe().to_string(index=False))

    def sample_from_prior(self, spin_measure: float, mass_measure: float) -> pd.DataFrame:
        """
        Search for possible samples in the prior.

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

        if self._is_mass_injected:
            # Find the possible samples in the prior
            # Based on:
            #    1. mass_prior - tol < mass_measure < mass_prior + tol
            #    2. spin_prior - tol < spin_measure < spin_prior + tol
            possible_samples = self._prior.loc[
                ((self._prior["mf_"] - mass_measure).abs() < self._mass_tolerance)
                & ((self._prior["chif"] - spin_measure).abs() < self._spin_tolerance)
            ]

            # Sample n_sample samples from the possible samples
            samples = self._sample_from_possible_samples(possible_samples)
        else:
            # Find the possible samples in the prior
            # Based on:
            #    1. spin_prior - tol < spin_measure < spin_prior + tol
            possible_samples = self._prior.loc[((self._prior["chif"] - spin_measure).abs() < self._spin_tolerance)]

            # Sample n_sample samples from the possible samples
            samples = self._sample_from_possible_samples(possible_samples)

            # Calculate the mass parameters (for mass not injected case)
            samples["m1"] = mass_measure / samples["mf"] * samples["q"] / (1 + samples["q"])
            samples["m2"] = mass_measure / samples["mf"] / (1 + samples["q"])
            samples["mf_"] = mass_measure

        return samples

    def _sample_from_possible_samples(self, df: pd.DataFrame) -> pd.DataFrame:
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
        elif len(df) < self._n_sample:
            local_logger.warning("Not enough similar samples in the prior.")
        else:
            df = df.sample(self._n_sample)
        return df
