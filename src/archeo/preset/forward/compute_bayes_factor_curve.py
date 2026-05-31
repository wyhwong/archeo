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
    """Compute Bayes-factor curve values across host escape velocities.

    Args:
        df_prior (pd.DataFrame): Baseline prior samples.
        df_posterior (pd.DataFrame): Posterior samples to reweight.
        df_bh1_binaries (pd.DataFrame): Candidate-prior dataframe for first-parent
            black holes. Must include ``v_esc``.
        df_bh2_binaries (pd.DataFrame): Candidate-prior dataframe for second-parent
            black holes. Must include ``v_esc``.
        n_workers (int): Number of worker processes used for curve sampling.

    Returns:
        pd.DataFrame: Tabular Bayes-factor curve with confidence interval bounds,
        median values, and curve metadata for each escape velocity.
    """

    n_workers = get_n_workers(n_workers)

    candidate_prior = CandidatePrior(df_bh1=df_bh1_binaries, df_bh2=df_bh2_binaries)
    bayes_factor_curve = BayesFactorCurve()
    bayes_factor_curve_data = bayes_factor_curve.get_bayes_factor_over_escape_velocity(
        prior=df_prior, posterior=df_posterior, candidate_prior=candidate_prior, n_workers=n_workers
    )
    return convert_bayes_factor_curve_to_dataframe(bayes_factor_curve_data, bayes_factor_curve.metadata)
