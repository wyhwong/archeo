import pandas as pd

import archeo.logger
from archeo.constants import Columns as C


local_logger = archeo.logger.get_logger(__name__)


class Prior(pd.DataFrame):
    """A class to represent the prior distribution."""

    def __init__(
        self,
        *args,
        is_mass_injected: bool = False,
        n_sample: int = 1,
        spin_tolerance: float = 0.05,  # unit: dimensionless
        mass_tolerance: float = 1.0,  # unit: solar mass
        **kwargs,
    ) -> None:
        """Initialize the class.

        Args:
        -----
            is_mass_injected (bool):
                Whether the mass is injected

            n_sample (int):
                The number of samples to be sampled each time

            spin_tolerance (float):
                The tolerance of the spin

            mass_tolerance (float):
                The tolerance of the mass
        """

        super().__init__(*args, **kwargs)

        self._is_mass_injected = is_mass_injected
        self._n_sample = n_sample
        self._spin_tolerance = spin_tolerance
        self._mass_tolerance = mass_tolerance

        local_logger.info(
            "Constructed a prior sampler [n=%d]: mass injected: %s, spin tol: %.2f, mass tol: %.2f.",
            self._n_sample,
            self._is_mass_injected,
            self._spin_tolerance,
            self._mass_tolerance,
        )

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
        else:
            df = df.sample(self._n_sample, replace=True)
        return df

    def retrieve_samples(self, spin_measure: float, mass_measure: float) -> pd.DataFrame:
        """Retrieve the samples.

        Args:
        -----
            spin_measure (float):
                The measured spin

            mass_measure (float):
                The measured mass

        Returns:
        -----
            pd.DataFrame:
                The sampled dataframe
        """

        if self._is_mass_injected:
            # Find the possible samples in the prior
            # Based on:
            #    1. mass_prior - tol < mass_measure < mass_prior + tol
            #    2. spin_prior - tol < spin_measure < spin_prior + tol
            possible_samples = self._prior.loc[
                ((self[C.BH_MASS] - mass_measure).abs() < self._mass_tolerance)
                & ((self[C.SPIN] - spin_measure).abs() < self._spin_tolerance)
            ]
            likelihood = len(possible_samples) / len(self._prior)

            # Sample n_sample samples from the possible samples
            samples = self._sample_from_possible_samples(possible_samples)
            samples[C.LIKELIHOOD] = likelihood
        else:
            # Find the possible samples in the prior
            # Based on:
            #    1. spin_prior - tol < spin_measure < spin_prior + tol
            possible_samples = self.loc[(self[C.SPIN] - spin_measure).abs() < self._spin_tolerance]
            likelihood = len(possible_samples) / len(self._prior)

            # Sample n_sample samples from the possible samples
            samples = self._sample_from_possible_samples(possible_samples)

            # Calculate the mass parameters (for mass not injected case)
            samples[C.HEAVIER_BH_MASS] = (
                mass_measure / samples[C.RETAINED_MASS] * samples[C.MASS_RATIO] / (1 + samples[C.MASS_RATIO])
            )
            samples[C.LIGHTER_BH_MASS] = mass_measure / samples[C.RETAINED_MASS] / (1 + samples[C.MASS_RATIO])
            samples[C.BH_MASS] = mass_measure
            samples[C.LIKELIHOOD] = likelihood

        return samples

    @property
    def _constructor(self):
        """Return the constructor of the class."""
        return Prior

    @classmethod
    def from_feather(cls, path: str, **kwargs) -> "Prior":
        """Read the feather file."""

        return cls(pd.read_feather(path), **kwargs)

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "Prior":
        """Read the csv file."""

        return cls(pd.read_csv(path), **kwargs)

    @classmethod
    def from_parquet(cls, path: str, **kwargs) -> "Prior":
        """Read the parquet file."""

        return cls(pd.read_parquet(path), **kwargs)
