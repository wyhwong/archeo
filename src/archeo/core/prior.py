from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import archeo.logger
from archeo.constants import SPEED_OF_LIGHT
from archeo.constants import Columns as C
from archeo.constants import Prefixes as P
from archeo.constants import Suffixes as S
from archeo.core.simulator import Simulator
from archeo.preset import get_prior_config
from archeo.schema import PriorConfig
from archeo.utils.parallel import multithread_run


local_logger = archeo.logger.get_logger(__name__)


class Prior(pd.DataFrame):
    """A class to represent the prior distribution."""

    def __init__(
        self,
        *args,
        rescale_mass: bool = False,
        sample_ratio: int = 1,
        spin_tolerance: float = 0.05,  # unit: dimensionless
        mass_tolerance: float = 1.0,  # unit: solar mass
        **kwargs,
    ) -> None:
        """Construct a prior dataframe.

        Args:
            rescale_mass (bool): Whether to enable mass marginalization
            sample_ratio (int): The number of samples to be sampled each time
            spin_tolerance (float): The tolerance of the spin
            mass_tolerance (float): The tolerance of the mass
        """

        super().__init__(*args, **kwargs)

        self._rescale_mass = rescale_mass
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
            # Return a number of samples with nan values
            df = pd.DataFrame(index=range(self._sample_ratio), columns=df.columns)
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

        if not self._rescale_mass:
            # Find the possible samples in the prior
            # Based on:
            #    1. mass_prior - tol < mass_measure < mass_prior + tol
            #    2. spin_prior - tol < spin_measure < spin_prior + tol
            possible_samples = self.loc[
                ((self[S.FINAL(C.MASS)] - mass_measure).abs() < self._mass_tolerance)
                & ((self[S.FINAL(C.SPIN_MAG)] - spin_measure).abs() < self._spin_tolerance)
            ]
            likelihood = len(possible_samples) / len(self)

            # Sample n_sample samples from the possible samples
            samples = self._sample_from_possible_samples(possible_samples)

            samples[C.LIKELIHOOD] = likelihood
            samples[P.ORIGINAL(S.FINAL(C.SPIN_MAG))] = spin_measure
            samples[P.ORIGINAL(S.FINAL(C.MASS))] = mass_measure

            return samples

        # Find the possible samples in the prior
        # Based on:
        #    1. spin_prior - tol < spin_measure < spin_prior + tol
        possible_samples = self.loc[(self[S.FINAL(C.SPIN_MAG)] - spin_measure).abs() < self._spin_tolerance]
        likelihood = len(possible_samples) / len(self)

        # Sample n_sample samples from the possible samples
        samples = self._sample_from_possible_samples(possible_samples)

        # Calculate the mass parameters (for mass not injected case)
        samples[S.PRIMARY(C.MASS)] = (
            mass_measure / samples[S.RETAINED(C.MASS)] * samples[C.MASS_RATIO] / (1 + samples[C.MASS_RATIO])
        )
        samples[S.SECONDARY(C.MASS)] = mass_measure / samples[S.RETAINED(C.MASS)] / (1 + samples[C.MASS_RATIO])
        samples[S.FINAL(C.MASS)] = mass_measure

        samples[C.LIKELIHOOD] = likelihood
        samples[P.ORIGINAL(S.FINAL(C.SPIN_MAG))] = spin_measure
        samples[P.ORIGINAL(S.FINAL(C.MASS))] = mass_measure

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

        # Define dummy columns
        df[C.RECOVERY_RATE] = float("nan")
        df[C.KS_TEST_FOR_MASS] = float("nan")
        df[C.KS_PV_FOR_MASS] = float("nan")
        df[C.KS_TEST_FOR_SPIN] = float("nan")
        df[C.KS_PV_FOR_SPIN] = float("nan")
        df[C.SAMPLE_ID] = float("nan")

        # Calculate the mass ratio
        m1, m2 = df[S.PRIMARY(C.MASS)], df[S.SECONDARY(C.MASS)]
        df[C.MASS_RATIO] = q = m1 / m2

        # Calculate the remnant mass
        df[S.FINAL(C.MASS)] = df[S.RETAINED(C.MASS)] * (m1 + m2)

        # Calculate the BH kick velocity
        df[C.KICK] = df[S.FINAL(C.VELOCITY)].apply(lambda vf: np.sqrt(np.dot(vf, vf)) * SPEED_OF_LIGHT)

        # Calculate the BH spin
        df[S.FINAL(C.SPIN_MAG)] = df[S.FINAL(C.SPIN)].apply(lambda vf: np.sqrt(np.dot(vf, vf)))

        # Calculate the parental spins
        df[S.PRIMARY(C.SPIN_MAG)] = df[S.PRIMARY(C.SPIN)].apply(lambda chi: np.sqrt(np.dot(chi, chi)))
        df[S.SECONDARY(C.SPIN_MAG)] = df[S.SECONDARY(C.SPIN)].apply(lambda chi: np.sqrt(np.dot(chi, chi)))

        # Calculate the effective spin
        a1z = df[S.PRIMARY(C.SPIN)].apply(lambda chi: chi[-1])
        a2z = df[S.SECONDARY(C.SPIN)].apply(lambda chi: chi[-1])
        df[S.EFF(C.SPIN)] = (m1 * a1z + m2 * a2z) / (m1 + m2)

        # Calculate the precession spin
        a1h = df[S.PRIMARY(C.SPIN)].apply(lambda chi: np.sqrt(chi[0] ** 2 + chi[1] ** 2))
        a2h = df[S.SECONDARY(C.SPIN)].apply(lambda chi: np.sqrt(chi[0] ** 2 + chi[1] ** 2))
        df[S.PREC(C.SPIN)] = np.maximum(a1h, (4 / q + 3) / (3 / q + 4) / q * a2h)

        return df

    def to_posterior(
        self,
        mass_posterior: list[float],
        spin_posterior: list[float],
        use_threads=True,
        n_threads: Optional[int] = None,
    ) -> pd.DataFrame:
        """Convert the prior to the posterior.

        Args:
            mass_posterior (list[float]): The posterior mass.
            spin_posterior (list[float]): The posterior spin.
            use_threads (bool): Whether to use threads.
            n_threads (Optional[int]): The number of threads to be used.

        Returns:
            pd.DataFrame: The posterior distribution.
        """

        if use_threads:
            input_kwargs = [
                dict(spin_measure=spin_measure, mass_measure=mass_measure)
                for spin_measure, mass_measure in zip(spin_posterior, mass_posterior)
            ]
            samples = multithread_run(
                func=self.retrieve_samples,
                input_kwargs=input_kwargs,
                n_threads=n_threads,
            )

        else:
            samples = [
                self.retrieve_samples(spin_measure=spin_measure, mass_measure=mass_measure)
                for spin_measure, mass_measure in zip(spin_posterior, mass_posterior)
            ]

        df_posterior = pd.concat(samples)

        df_posterior[C.RECOVERY_RATE] = df_posterior[C.KICK].notna().sum() / len(df_posterior)

        ks, p_value = ks_2samp(df_posterior[P.ORIGINAL(S.FINAL(C.SPIN_MAG))], df_posterior[S.FINAL(C.SPIN_MAG)])
        df_posterior[C.KS_TEST_FOR_SPIN] = ks
        df_posterior[C.KS_PV_FOR_SPIN] = p_value

        ks, p_value = ks_2samp(df_posterior[P.ORIGINAL(S.FINAL(C.MASS))], df_posterior[S.FINAL(C.MASS)])
        df_posterior[C.KS_TEST_FOR_MASS] = ks
        df_posterior[C.KS_PV_FOR_MASS] = p_value

        df_posterior[C.SAMPLE_ID] = df_posterior.apply(lambda x: x.name if pd.notna(x[C.KICK]) else None, axis=1)
        df_posterior = df_posterior.reset_index(drop=True)

        return df_posterior
