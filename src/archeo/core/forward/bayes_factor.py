from typing import Union

import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt

from archeo.core.forward.resampler import ImportanceSamplingData as ISData
from archeo.utils.helper import pre_release
from archeo.utils.parallel import multithread_run


@retry(stop=stop_after_attempt(3))
def _resample_and_get_bayes_factor(
    candidate_prior: pd.DataFrame, prior: pd.DataFrame, posterior: pd.DataFrame
) -> float:
    """Resample the prior and posterior samples and compute the Bayes factor."""

    data = ISData(
        new_prior_samples=candidate_prior.sample(len(candidate_prior), replace=True),
        posterior_samples=posterior.sample(len(posterior), replace=True),
        prior_samples=prior.sample(len(prior), replace=True),
    )
    return data.get_bayes_factor()


def _extract_kick_thresholds(
    kicks: Union[list[pd.Series], pd.Series],
) -> tuple[float, float]:
    """Extract the lower and upper bounds of the kick thresholds."""

    if isinstance(kicks, pd.Series):
        # Here we ensure at least 1000 samples for the kick threshold
        k_lb = kicks.quantile(1000 / len(kicks))
        k_ub = kicks.max()
    else:
        # Here we ensure at least 1000 samples for each kick threshold
        k_lb = max([ks.quantile(1000 / len(ks)) for ks in kicks])
        k_ub = max([ks.max() for ks in kicks])

    return k_lb, k_ub


@pre_release
def get_bayes_factor_over_escape_velocity(
    prior: pd.DataFrame,
    posterior: pd.DataFrame,
    candidate_prior: dict[str, pd.Series],
    kicks: Union[list[pd.Series], pd.Series],
    n_trials: int = 500,
    ref_bayes_factor: float = 1.0,
) -> dict[str, Union[list[float], list[list[float]]]]:
    """Compute the Bayes factor over a range of kick thresholds.

    Args:
        prior (pd.DataFrame): Prior samples.
        posterior (pd.DataFrame): Posterior samples.
        candidate_prior (dict[str, pd.Series]): Candidate prior samples.
        kicks (Union[list[pd.Series], pd.Series]): Kick thresholds.
        n_trials (int): Number of trials to run for resampling.
        ref_bayes_factor (float): Reference Bayes factor to normalize the results.

    Returns:
        dict: A dictionary containing the kick cutoff values, Bayes factor samples,
              median Bayes factor, and the 5th and 95th percentiles of the Bayes factor samples.
    """

    k_lb, k_ub = _extract_kick_thresholds(kicks)
    _kicks = (
        # Sample 100 evenly spaced values in log space between k_lb and k_ub
        np.logspace(np.log10(k_lb), np.log10(k_ub), 100)
        .astype(float)
        .tolist()
    )
    _bf_samples: list[list[float]] = []

    for k in _kicks:
        # Mark index of the samples that are below the kick threshold
        if isinstance(kicks, pd.Series):
            idx = kicks < k
            _candidate_prior = {col: ds[idx] for col, ds in candidate_prior.items()}
        else:
            _candidate_prior = {}
            for col, ds in candidate_prior.items():
                idx = kicks[col.endswith("2")] < k
                _candidate_prior[col] = ds[idx]

        _bfs = multithread_run(
            func=_resample_and_get_bayes_factor,
            input_kwargs=[
                {
                    "candidate_prior": pd.DataFrame(_candidate_prior),
                    "prior": prior,
                    "posterior": posterior,
                }
                for _ in range(n_trials)
            ],
        )

        _bfs = [_bf / ref_bayes_factor for _bf in _bfs]
        _bf_samples.append(_bfs)

    return {
        "kick_cutoff": _kicks,
        "bayes_factor_samples": _bf_samples,
        "bayes_factor_median": [np.median(bfs) for bfs in _bf_samples],
        "bayes_factor_low": [np.percentile(bfs, q=5) for bfs in _bf_samples],
        "bayes_factor_high": [np.percentile(bfs, q=95) for bfs in _bf_samples],
    }
