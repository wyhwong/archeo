from typing import Union

import numpy as np
import pandas as pd

import archeo.logger
from archeo.constants import SPEED_OF_LIGHT
from archeo.constants import Columns as C
from archeo.core.simulator import Simulator
from archeo.preset import get_prior_config
from archeo.schema import PriorConfig
from archeo.utils.executor import MultiThreadExecutor


local_logger = archeo.logger.get_logger(__name__)


class Prior(pd.DataFrame):
    """A class to represent the prior distribution."""

    def __init__(
        self,
        *args,
        is_mass_marginalized: bool = False,
        sample_ratio: int = 1,
        spin_tolerance: float = 0.05,  # unit: dimensionless
        mass_tolerance: float = 1.0,  # unit: solar mass
        **kwargs,
    ) -> None:
        """Construct a prior dataframe.

        Args:
            is_mass_marginalized (bool): Whether to enable mass marginalization
            sample_ratio (int): The number of samples to be sampled each time
            spin_tolerance (float): The tolerance of the spin
            mass_tolerance (float): The tolerance of the mass
        """

        super().__init__(*args, **kwargs)

        self._is_mass_marginalized = is_mass_marginalized
        self._sample_ratio = sample_ratio
        self._spin_tolerance = spin_tolerance
        self._mass_tolerance = mass_tolerance

    def _sample_from_possible_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample from a dataframe.

        Args:
            df (pd.DataFrame): The dataframe to sample from.

        Returns:
            df (pd.DataFrame): The sampled dataframe.
        """

        if df.empty:
            local_logger.warning("No similar samples in the prior.")
        else:
            df = df.sample(self._sample_ratio, replace=True)
        return df

    def retrieve_samples(self, spin_measure: float, mass_measure: float) -> pd.DataFrame:
        """Retrieve the samples from prior.

        Args:
            spin_measure (float): The measured spin
            mass_measure (float): The measured mass

        Returns:
            pd.DataFrame: The sampled dataframe
        """

        if not self._is_mass_marginalized:
            # Find the possible samples in the prior
            # Based on:
            #    1. mass_prior - tol < mass_measure < mass_prior + tol
            #    2. spin_prior - tol < spin_measure < spin_prior + tol
            possible_samples = self.loc[
                ((self[C.BH_MASS] - mass_measure).abs() < self._mass_tolerance)
                & ((self[C.BH_SPIN] - spin_measure).abs() < self._spin_tolerance)
            ]
            likelihood = len(possible_samples) / len(self)

            # Sample n_sample samples from the possible samples
            samples = self._sample_from_possible_samples(possible_samples)
            samples[C.LIKELIHOOD] = likelihood
            return samples

        # Find the possible samples in the prior
        # Based on:
        #    1. spin_prior - tol < spin_measure < spin_prior + tol
        possible_samples = self.loc[(self[C.BH_SPIN] - spin_measure).abs() < self._spin_tolerance]
        likelihood = len(possible_samples) / len(self)

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

        return pd.DataFrame

    @classmethod
    def from_feather(cls, path: str, **kwargs) -> "Prior":
        """Read the feather file.

        Args:
            path (str): The path to the feather file.
            **kwargs: The keyword arguments for parental class (pd.DataFrame).

        Returns:
            Prior: The prior distribution.
        """

        return cls(pd.read_feather(path), **kwargs)

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "Prior":
        """Read the csv file.

        Args:
            path (str): The path to the csv file.
            **kwargs: The keyword arguments for parental class (pd.DataFrame).

        Returns:
            Prior: The prior distribution
        """

        return cls(pd.read_csv(path), **kwargs)

    @classmethod
    def from_json(cls, path: str, **kwargs) -> "Prior":
        """Read the json file.

        Args:
            path (str): The path to the json file.
            **kwargs: The keyword arguments for parental class (pd.DataFrame).

        Returns:
            Prior: The prior distribution.
        """

        return cls(pd.read_json(path), **kwargs)

    @classmethod
    def from_parquet(cls, path: str, **kwargs) -> "Prior":
        """Read the parquet file.

        Args:
            path (str): The path to the parquet file.
            **kwargs: The keyword arguments for parental class (pd.DataFrame).

        Returns:
            Prior: The prior distribution.
        """

        return cls(pd.read_parquet(path), **kwargs)

    @classmethod
    def from_config(cls, prior_config: Union[PriorConfig, str], use_threads=True, **kwargs) -> "Prior":
        """Generate the prior from the prior config.

        Args:
            prior_config (PriorConfig): The prior configuration.
            use_threads (bool): Whether to use threads.
            **kwargs: The keyword arguments for the class.

        Returns:
            Prior: The prior distribution.
        """

        if isinstance(prior_config, str):
            prior_config = get_prior_config(prior_config)

        simulator = Simulator(prior_config)
        return cls.from_simulator(simulator, use_threads=use_threads, **kwargs)

    @classmethod
    def from_simulator(cls, simulator: Simulator, use_threads=True, **kwargs) -> "Prior":
        """Generate the prior from the simulator.

        Args:
            simulator (Simulator): The simulator.
            use_threads (bool): Whether to use threads.
            **kwargs: The keyword arguments for the class.

        Returns:
            Prior: The prior distribution.
        """

        df = cls(simulator.simulate(use_threads=use_threads), **kwargs)

        # Extract more information from the samples

        # Define nan recovery rate
        df[C.RECOVERY_RATE] = float("nan")

        # Calculate the mass ratio
        m1, m2 = df[C.HEAVIER_BH_MASS], df[C.LIGHTER_BH_MASS]
        df[C.MASS_RATIO] = q = m1 / m2

        # Calculate the remnant mass
        df[C.BH_MASS] = df[C.RETAINED_MASS] * (m1 + m2)

        # Calculate the BH kick velocity
        df[C.BH_KICK] = df[C.BH_VEL].apply(lambda vf: np.sqrt(np.dot(vf, vf)) * SPEED_OF_LIGHT)

        # Calculate the BH spin
        df[C.BH_SPIN] = df[C.BH_CHI].apply(lambda vf: np.sqrt(np.dot(vf, vf)))

        # Calculate the parental spins
        df[C.HEAVIER_BH_SPIN] = df[C.HEAVIER_BH_CHI].apply(lambda chi: np.sqrt(np.dot(chi, chi)))
        df[C.LIGHTER_BH_SPIN] = df[C.LIGHTER_BH_CHI].apply(lambda chi: np.sqrt(np.dot(chi, chi)))

        # Calculate the effective spin
        a1z = df[C.HEAVIER_BH_CHI].apply(lambda chi: chi[-1])
        a2z = df[C.LIGHTER_BH_CHI].apply(lambda chi: chi[-1])
        df[C.BH_EFF_SPIN] = (m1 * a1z + m2 * a2z) / (m1 + m2)

        # Calculate the precession spin
        a1h = df[C.HEAVIER_BH_CHI].apply(lambda chi: np.sqrt(chi[0] ** 2 + chi[1] ** 2))
        a2h = df[C.LIGHTER_BH_CHI].apply(lambda chi: np.sqrt(chi[0] ** 2 + chi[1] ** 2))
        df[C.BH_PREC_SPIN] = np.maximum(a1h, (4 / q + 3) / (3 / q + 4) / q * a2h)

        return df

    def to_posterior(
        self,
        mass_posterior: list[float],
        spin_posterior: list[float],
        use_threads=True,
    ) -> pd.DataFrame:
        """Convert the prior to the posterior.

        Args:
            mass_posterior (list[float]): The posterior mass.
            spin_posterior (list[float]): The posterior spin.
            use_threads (bool): Whether to use threads.

        Returns:
            pd.DataFrame: The posterior distribution.
        """

        if use_threads:
            exc = MultiThreadExecutor()
            input_kwargs = [
                dict(spin_measure=spin_measure, mass_measure=mass_measure)
                for spin_measure, mass_measure in zip(spin_posterior, mass_posterior)
            ]
            samples = exc.run(func=self.retrieve_samples, input_kwargs=input_kwargs)

        else:
            samples = [
                self.retrieve_samples(spin_measure=spin_measure, mass_measure=mass_measure)
                for spin_measure, mass_measure in zip(spin_posterior, mass_posterior)
            ]

        df_posterior = pd.concat(samples)
        df_posterior[C.RECOVERY_RATE] = len(df_posterior) / (len(mass_posterior) * self._sample_ratio)

        return df_posterior
