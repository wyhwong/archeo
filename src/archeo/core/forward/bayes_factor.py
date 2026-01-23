from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from archeo.core.forward.resampler import ImportanceSamplingData as ISData
from archeo.utils.helper import pre_release


def _extract_kick_thresholds(
    kicks: Union[list[pd.Series], pd.Series], n_bins: int = 50, log_scale: bool = True
) -> list[float]:
    """Extract the lower and upper bounds of the kick thresholds."""

    if isinstance(kicks, pd.Series):
        # Here we ensure at least 1 samples for the kick threshold
        k_lb = kicks.quantile(1 / len(kicks))
        k_ub = kicks.max()
    else:
        # Here we ensure at least 1 samples for each kick threshold
        k_lb = max([ks.quantile(1 / len(ks)) for ks in kicks if ks.min() > 0])
        k_ub = max([ks.max() for ks in kicks])

    if log_scale:
        ks = np.logspace(np.log10(k_lb), np.log10(k_ub), n_bins).astype(float)
    else:
        ks = np.linspace(k_lb, k_ub, n_bins).astype(float)

    return ks.tolist()


def _prepare_new_prior_samples_for_v_esc(
    new_prior: pd.DataFrame,
    kicks: pd.Series,
    v_esc: float,
    least_sample_count: int = 10000,
) -> None:

    new_prior = new_prior.loc[kicks <= v_esc]
    if len(new_prior) < least_sample_count:
        new_prior = pd.concat(
            [new_prior, new_prior.sample(least_sample_count - len(new_prior), replace=True)],
            ignore_index=True,
        )
    return new_prior


@dataclass
class BayesFactorConfig:
    """
    Data class for Bayes factor configuration.

    n_bootstrapping (int): Number of trials to run for resampling.
    n_bins (int): Number of bins to use for kick thresholds.
    is_kick_in_log_scale (bool): Whether to use log scale for escape velocity thresholds.
    ref_bayes_factor (float): Reference Bayes factor to normalize the results.
    binsize_spin (float): Bin size for spin parameter.
    binsize_mass (float): Bin size for mass parameter.
    assume_parameter_independence (bool): Whether to assume parameter independence.
    """

    n_bootstrapping: int = 50
    n_bins: int = 50
    is_kick_in_log_scale: bool = True
    least_new_prior_sample_count: int = 10000
    ref_bayes_factor: float = 1.0
    binsize_spin: float = 0.05
    binsize_mass: float = 1.0
    assume_parameter_independence: bool = False


@pre_release
def get_bayes_factor_over_escape_velocity(
    prior: pd.DataFrame,
    posterior: pd.DataFrame,
    new_prior: pd.DataFrame,
    kicks: Union[list[pd.Series], pd.Series],
    bfc: BayesFactorConfig = BayesFactorConfig(),
) -> dict[str, Union[list[float], list[list[float]]]]:
    """Compute the Bayes factor over a range of kick thresholds.

    Args:
        prior (pd.DataFrame): Prior samples.
        posterior (pd.DataFrame): Posterior samples.
        new_prior (pd.DataFrame): New prior samples.
        kicks (Union[list[pd.Series], pd.Series]): Escape velocity of new prior samples.
        bfc (BayesFactorConfig): Configuration for Bayes factor computation.

    Returns:
        dict: A dictionary containing the kick cutoff values, Bayes factor samples,
              median Bayes factor, and the 5th and 95th percentiles of the Bayes factor samples.
    """

    v_escs = _extract_kick_thresholds(kicks, n_bins=bfc.n_bins, log_scale=bfc.is_kick_in_log_scale)
    _bf_samples: list[list[float]] = []

    for v_esc in v_escs:
        # Mark index of the samples that are below the kick threshold
        if isinstance(kicks, pd.Series):
            _new_prior = _prepare_new_prior_samples_for_v_esc(
                new_prior=new_prior,
                kicks=kicks,
                v_esc=v_esc,
                least_sample_count=bfc.least_new_prior_sample_count,
            )
        else:
            _new_prior = pd.concat(
                [
                    _prepare_new_prior_samples_for_v_esc(
                        new_prior=new_prior[[c for c in new_prior.columns if "1" in c]],
                        kicks=kicks[0],
                        v_esc=v_esc,
                        least_sample_count=bfc.least_new_prior_sample_count,
                    ),
                    _prepare_new_prior_samples_for_v_esc(
                        new_prior=new_prior[[c for c in new_prior.columns if "2" in c]],
                        kicks=kicks[1],
                        v_esc=v_esc,
                        least_sample_count=bfc.least_new_prior_sample_count,
                    ),
                ],
                axis=1,
            )

        _bfs = ISData(
            prior_samples=prior,
            posterior_samples=posterior,
            new_prior_samples=_new_prior,
            binsize_spin=bfc.binsize_spin,
            binsize_mass=bfc.binsize_mass,
            assume_parameter_independence=bfc.assume_parameter_independence,
        ).sample_bayes_factor(n=bfc.n_bootstrapping, is_parallel=True)

        _bfs = [_bf / bfc.ref_bayes_factor for _bf in _bfs]
        _bf_samples.append(_bfs)

    return {
        "escape_velocity": v_escs,
        "bayes_factor_samples": _bf_samples,
        "bayes_factor_median": [np.median(bfs) for bfs in _bf_samples],
        "bayes_factor_low": [np.percentile(bfs, q=5) for bfs in _bf_samples],
        "bayes_factor_high": [np.percentile(bfs, q=95) for bfs in _bf_samples],
    }
