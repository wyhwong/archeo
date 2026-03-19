import pandas as pd

from archeo.bayesian.importance_sampling import BayesFactorCurve, CandidatePrior
from archeo.postprocessing.dataframe import convert_bayes_factor_curve_to_dataframe
from archeo.utils.parallel import get_n_workers


def compute_bayes_factor_curve_over_escape_velocity(
    df_prior: pd.DataFrame,
    df_posterior: pd.DataFrame,
    df_bh1_binaries: pd.DataFrame,
    df_bh2_binaries: pd.DataFrame,
    n_workers: int = 1,
) -> pd.DataFrame:
    """Compute the Bayes factor curve for the given prior and posterior distributions."""

    n_workers = get_n_workers(n_workers)

    candidate_prior = CandidatePrior(df_bh1=df_bh1_binaries, df_bh2=df_bh2_binaries)
    bayes_factor_curve = BayesFactorCurve()
    bayes_factor_curve_data = bayes_factor_curve.get_bayes_factor_over_escape_velocity(
        prior=df_prior, posterior=df_posterior, candidate_prior=candidate_prior, n_workers=n_workers
    )
    return convert_bayes_factor_curve_to_dataframe(bayes_factor_curve_data, bayes_factor_curve.metadata)
