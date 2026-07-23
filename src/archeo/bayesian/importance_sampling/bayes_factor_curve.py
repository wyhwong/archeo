from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from archeo.bayesian.importance_sampling.resampler.interface import ImportanceSamplingData as ISData
from archeo.data_structures.bayesian.bayes_factor import BayesFactor, BayesFactorCurveData, BayesFactorCurveMetadata
from archeo.utils.decorator import pre_release
from archeo.utils.parallel import multiprocess_run


class CandidatePrior(BaseModel):
    """Container for candidate-prior components used in Bayes-factor curves.

    Args:
        df_bh1 (pd.DataFrame): Candidate prior component for first parent BH.
        df_bh2 (pd.DataFrame): Candidate prior component for second parent BH.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    df_bh1: pd.DataFrame
    df_bh2: pd.DataFrame

    # Validation:
    # 1. Check df_bh1 and df_bh2 do not have any overlapping columns except for "v_esc"
    # 2. Check df_bh1 and df_bh2 have "v_esc" column
    @model_validator(mode="after")
    @classmethod
    def validate_dataframes(cls, values):
        """Validate candidate-prior dataframe compatibility.

        Args:
            values: CandidatePrior instance under validation.

        Returns:
            CandidatePrior: Validated instance.

        Raises:
            ValueError: If ``v_esc`` is missing in either dataframe or if non-``v_esc``
            overlapping columns are present.
        """

        if ("v_esc" not in values.df_bh1.columns) or ("v_esc" not in values.df_bh2.columns):
            raise ValueError("Both df_bh1 and df_bh2 must have 'v_esc' column.")

        overlapping_columns = set(values.df_bh1.columns).intersection(set(values.df_bh2.columns)) - {"v_esc"}
        if overlapping_columns:
            raise ValueError(f"df_bh1 and df_bh2 have overlapping columns: {overlapping_columns}")

        return values

    def get_conditional_prior(
        self, v_esc: float, n_min: int = 500000, random_state: Optional[int] = None
    ) -> pd.DataFrame:
        """Build conditional candidate prior at an escape-velocity threshold.

        Args:
            v_esc (float): Maximum host escape velocity retained.
            n_min (int): Minimum number of sampled rows per side before concatenation.
            random_state (Optional[int]): Seed for reproducible resampling.

        Returns:
            pd.DataFrame: Concatenated conditional prior dataframe for the threshold.
        """

        df_bh1_prior = self.df_bh1.loc[self.df_bh1["v_esc"] <= v_esc]
        df_bh2_prior = self.df_bh2.loc[self.df_bh2["v_esc"] <= v_esc]

        if df_bh1_prior.empty or df_bh2_prior.empty:
            return pd.DataFrame(columns=self.df_bh1.columns.union(self.df_bh2.columns))

        n_samples = max(len(df_bh1_prior), len(df_bh2_prior), n_min)
        seed_sequence = np.random.SeedSequence(random_state)
        random_states = seed_sequence.generate_state(2)
        conditional_prior = pd.concat(
            [
                df_bh1_prior.sample(n=n_samples, replace=True, random_state=random_states[0])
                .drop(columns=["v_esc"])
                .reset_index(drop=True),
                df_bh2_prior.sample(n=n_samples, replace=True, random_state=random_states[1])
                .drop(columns=["v_esc"])
                .reset_index(drop=True),
            ],
            axis=1,
        )
        return conditional_prior

    def get_host_escape_velocities(self, n_pts: int = 50, log_scale: bool = True) -> list[float]:
        """Generate evaluation grid of escape velocities.

        Args:
            n_pts (int): Number of velocity grid points before optional extension.
            log_scale (bool): If ``True``, use logarithmic spacing; otherwise linear.

        Returns:
            list[float]: Escape-velocity grid, with an added terminal 5000 value when
            not already covered.
        """

        v_esc_min = min(self.df_bh1["v_esc"].min(), self.df_bh2["v_esc"].min())
        v_esc_max = max(self.df_bh1["v_esc"].max(), self.df_bh2["v_esc"].max())

        if log_scale:
            v_escs = np.logspace(max(np.log10(v_esc_min), np.log10(0.1)), np.log10(v_esc_max), n_pts)
        else:
            v_escs = np.linspace(v_esc_min, v_esc_max, n_pts)

        # Add 5000 km/s in the end of the list if it's not already included
        if v_escs[-1] < 5000:
            v_escs = np.append(v_escs, 5000.0)

        return v_escs.tolist()


class BayesFactorCurve(BaseModel, frozen=True):
    """Data class for Bayes factor curve."""

    n_bootstrapping: int = 50
    n_pts: int = 10
    log_scale: bool = False
    metadata: BayesFactorCurveMetadata = BayesFactorCurveMetadata()

    def _sample_bayes_factor(
        self,
        prior: pd.DataFrame,
        posterior: pd.DataFrame,
        candidate_prior: CandidatePrior,
        v_esc: float,
        random_state: Optional[int] = None,
    ) -> BayesFactor:
        """Sample Bayes-factor bootstrap distribution at one velocity value.

        Args:
            prior (pd.DataFrame): Baseline prior samples.
            posterior (pd.DataFrame): Posterior samples.
            candidate_prior (CandidatePrior): Candidate prior provider.
            v_esc (float): Escape velocity threshold.
            random_state (Optional[int]): Random seed for reproducibility.

        Returns:
            BayesFactor: Bootstrap Bayes-factor samples.
        """

        return ISData(
            prior_samples=prior,
            posterior_samples=posterior,
            new_prior_samples=candidate_prior.get_conditional_prior(v_esc, random_state=random_state),
            binsize_spin=self.metadata.binsize_spin,
            binsize_mass=self.metadata.binsize_mass,
            assume_parameter_independence=self.metadata.assume_parameter_independence,
        ).sample_bayes_factor(n=self.n_bootstrapping, is_parallel=True)

    @pre_release
    def get_bayes_factor_over_escape_velocity(
        self,
        prior: pd.DataFrame,
        posterior: pd.DataFrame,
        candidate_prior: CandidatePrior,
        random_state: Optional[int] = None,
        n_workers: int = 1,
    ) -> BayesFactorCurveData:
        """Compute Bayes factors over a grid of host escape velocities.

        Args:
            prior (pd.DataFrame): Baseline prior samples.
            posterior (pd.DataFrame): Posterior samples.
            candidate_prior (CandidatePrior): Candidate prior provider.
            random_state (Optional[int]): Random seed for reproducibility.
            n_workers (int): Number of worker processes.

        Returns:
            BayesFactorCurveData: Mapping from escape velocity to sampled Bayes factor.
        """

        v_escs = candidate_prior.get_host_escape_velocities(n_pts=self.n_pts, log_scale=self.log_scale)
        seed_sequence = np.random.SeedSequence(random_state)

        if n_workers == 1:
            random_states = seed_sequence.generate_state(len(v_escs))
            return {
                v_esc: self._sample_bayes_factor(prior, posterior, candidate_prior, v_esc, _random_state)
                for _random_state, v_esc in zip(random_states, v_escs)
            }

        random_states = seed_sequence.generate_state(len(v_escs))
        bayes_factor_data = multiprocess_run(
            func=self._sample_bayes_factor,
            input_kwargs=[
                {
                    "prior": prior,
                    "posterior": posterior,
                    "candidate_prior": candidate_prior,
                    "v_esc": v_esc,
                    "random_state": random_states[i],
                }
                for i, v_esc in enumerate(v_escs)
            ],
            n_processes=n_workers,
        )
        return {v_esc: bayes_factor for v_esc, bayes_factor in zip(v_escs, bayes_factor_data)}
