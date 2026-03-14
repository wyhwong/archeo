import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from archeo.bayesian.importance_sampling.resampler.interface import ImportanceSamplingData as ISData
from archeo.data_structures.bayesian.bayes_factor import BayesFactor, BayesFactorCurveData, BayesFactorCurveMetadata
from archeo.utils.decorator import pre_release
from archeo.utils.parallel import multiprocess_run


class CandidatePrior(BaseModel):
    """Data class for candidate prior samples."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    df_bh1: pd.DataFrame
    df_bh2: pd.DataFrame

    # Validation:
    # 1. Check df_bh1 and df_bh2 do not have any overlapping colunms except for "v_esc"
    # 2. Check df_bh1 and df_bh2 have "v_esc" column
    @model_validator(mode="after")
    @classmethod
    def validate_dataframes(cls, values):

        if ("v_esc" not in values.df_bh1.columns) or ("v_esc" not in values.df_bh2.columns):
            raise ValueError("Both df_bh1 and df_bh2 must have 'v_esc' column.")

        overlapping_columns = set(values.df_bh1.columns).intersection(set(values.df_bh2.columns)) - {"v_esc"}
        if overlapping_columns:
            raise ValueError(f"df_bh1 and df_bh2 have overlapping columns: {overlapping_columns}")

        return values

    def get_conditional_prior(self, v_esc: float, n_min: int = 500000, random_state: int = 42) -> pd.DataFrame:
        """Get the prior samples for the given escape velocity threshold."""

        df_bh1_prior = self.df_bh1.loc[self.df_bh1["v_esc"] <= v_esc]
        df_bh2_prior = self.df_bh2.loc[self.df_bh2["v_esc"] <= v_esc]

        if df_bh1_prior.empty and df_bh2_prior.empty:
            return pd.DataFrame(columns=self.df_bh1.columns.union(self.df_bh2.columns))

        n_samples = max(len(df_bh1_prior), len(df_bh2_prior), n_min)
        conditional_prior = pd.concat(
            [
                df_bh1_prior.sample(n=n_samples, replace=True, random_state=random_state).reset_index(drop=True),
                df_bh2_prior.sample(n=n_samples, replace=True, random_state=random_state).reset_index(drop=True),
            ],
            axis=1,
        )
        return conditional_prior

    def get_host_escape_velocities(self, n_pts: int = 50, log_scale: bool = True) -> list[float]:

        v_esc_min = min(self.df_bh1["v_esc"].min(), self.df_bh2["v_esc"].min())
        v_esc_max = max(self.df_bh1["v_esc"].max(), self.df_bh2["v_esc"].max())

        if log_scale:
            v_escs = np.logspace(np.log10(v_esc_min), np.log10(v_esc_max), n_pts)
        else:
            v_escs = np.linspace(v_esc_min, v_esc_max, n_pts)

        # Add zero in the beginning of the list if it's not already included
        if v_escs[0] > 0:
            v_escs = np.insert(v_escs, 0, 0.0)

        # Add 5000 km/s in the end of the list if it's not already included
        if v_escs[-1] < 5000:
            v_escs = np.append(v_escs, 5000.0)

        return v_escs.tolist()


class BayesFactorCurve(BaseModel, frozen=True):
    """Data class for Bayes factor curve."""

    n_bootstrapping: int = 50
    n_pts: int = 10
    log_scale: bool = False
    assume_parameter_independence: bool = False
    metadata: BayesFactorCurveMetadata = BayesFactorCurveMetadata()

    def _sample_bayes_factor(
        self,
        prior: pd.DataFrame,
        posterior: pd.DataFrame,
        candidate_prior: CandidatePrior,
        v_esc: float,
    ) -> BayesFactor:

        return ISData(
            prior_samples=prior,
            posterior_samples=posterior,
            new_prior_samples=candidate_prior.get_conditional_prior(v_esc),
            binsize_spin=self.metadata.binsize_spin,
            binsize_mass=self.metadata.binsize_mass,
            assume_parameter_independence=self.assume_parameter_independence,
        ).sample_bayes_factor(n=self.n_bootstrapping, is_parallel=True)

    @pre_release
    def get_bayes_factor_over_escape_velocity(
        self,
        prior: pd.DataFrame,
        posterior: pd.DataFrame,
        candidate_prior: CandidatePrior,
        n_workers: int = 1,
    ) -> BayesFactorCurveData:

        v_escs = candidate_prior.get_host_escape_velocities(n_pts=self.n_pts, log_scale=self.log_scale)

        if n_workers == 1:
            return {v_esc: self._sample_bayes_factor(prior, posterior, candidate_prior, v_esc) for v_esc in v_escs}

        bayes_factor_data = multiprocess_run(
            func=self._sample_bayes_factor,
            input_kwargs=[
                {"prior": prior, "posterior": posterior, "candidate_prior": candidate_prior, "v_esc": v_esc}
                for v_esc in v_escs
            ],
            n_processes=n_workers,
        )
        return {v_esc: bayes_factor for v_esc, bayes_factor in zip(v_escs, bayes_factor_data)}
