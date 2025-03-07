from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rv_histogram

import archeo.logger
from archeo.schema import Domain
from archeo.utils.helper import pre_release


local_logger = archeo.logger.get_logger(__name__)


@dataclass
class ISData:
    """Importance sampling data"""

    posterior: pd.DataFrame
    prior: pd.DataFrame
    candidate_prior: dict[str, pd.Series]

    def _check_column_existence(self) -> None:
        """Check if the columns in the dataframes are consistent"""

        for col in self.candidate_prior.keys():
            if col not in self.prior.columns:
                raise ValueError(f"Column {col} is not in the prior dataframe")
            if col not in self.posterior.columns:
                raise ValueError(f"Column {col} is not in the posterior dataframe")

        for col in self.prior.columns:
            if col not in self.posterior.columns:
                raise ValueError(f"Prior column {col} is not in the posterior dataframe")

        for col in self.posterior.columns:
            if col not in self.prior.columns:
                raise ValueError(f"Posterior column {col} is not in the prior dataframe")

    def _compute_bounds(self) -> dict[str, Domain]:
        """Get the bounds of the dataframes"""

        bounds: dict[str, Domain] = {}

        for col in self.posterior.columns:
            _min = min(
                self.posterior[col].min(),
                self.prior[col].min(),
                (self.candidate_prior[col].min() if col in self.candidate_prior else np.inf),
            )
            _max = max(
                self.posterior[col].max(),
                self.prior[col].max(),
                (self.candidate_prior[col].max() if col in self.candidate_prior else -np.inf),
            )

            bounds[col] = Domain(low=_min, high=_max)

        return bounds

    def __post_init__(self) -> None:
        """Check if the columns in the dataframes are consistent"""

        self._check_column_existence()
        self._bounds = self._compute_bounds()

    @property
    def bounds(self) -> dict[str, Domain]:
        """Get the bounds of the dataframes"""

        return self._bounds

    @property
    def correlated_params(self) -> list[str]:
        """Get the correlated parameters"""

        return [col for col in self.prior.columns if col not in self.candidate_prior]


def get_histrogram(samples: pd.Series, bounds: Domain, nbins: int = 70, atol: float = 1e-6) -> np.ndarray:
    """Compute the histogram of the samples"""

    hist, edges = np.histogram(samples, bins=nbins, density=True, range=(bounds.low, bounds.high))
    binwidth = edges[1] - edges[0]

    auc = np.sum(hist) * binwidth
    if not np.isclose(auc, 1.0, atol=atol):
        msg = f"Invalid probability distribution (AUC={auc:.2f})."
        local_logger.error(msg)
        raise ValueError(msg)

    return hist


@pre_release
def get_weights(data: ISData, nbins=70, ztol=1e-8) -> np.ndarray:
    """Get the weights for the importance sampling

    Args:
        data (ISData): Importance sampling data.
        nbins (int, optional): Number of bins for the histogram. Defaults to 70.
        ztol (float, optional): Tolerance for the zero division. Defaults to 1e-8.

    Returns:
        weights: np.ndarray, the weights for the importance sampling.
    """

    weights = np.ones(len(data.posterior))

    for col, samples in data.candidate_prior.items():
        prior_hist = get_histrogram(data.prior[col], data.bounds[col], nbins)
        candidate_prior_hist = get_histrogram(samples, data.bounds[col], nbins)

        # Avoid division by zero
        ratio = np.where(prior_hist > ztol, np.exp(np.log(candidate_prior_hist) - np.log(prior_hist)), 0.0)
        rv = rv_histogram((ratio, np.linspace(data.bounds[col].low, data.bounds[col].high, nbins + 1)))
        weights *= rv.pdf(data.posterior[col])

    return weights


@pre_release
def get_bayes_factor(data: ISData, nbins=70, random_state=42, ztol: float = 1e-8) -> float:
    """Compute the Bayes factor between two models

    NOTE: In this implementation, the likelihood function remains untouched.
    So that the Bayes factor is computed as the ratio of the new prior to the old prior.
    Details please check importance sampling.

    Args:
        data (ISData): Importance sampling data.
        nbins (int, optional): Number of bins for the histogram. Defaults to 70.
        random_state (int, optional): Random state for resampling the prior. Defaults to 42.
        ztol (float, optional): Tolerance for the zero division. Defaults to 1e-8.

    Returns:
        bf: float, the Bayes factor.
    """

    bf = 1.0
    weights = np.ones(len(data.prior))

    # Here we first loop over the candidate prior columns
    # We need to compute the weights for resampling the prior
    # for the correction of potentially correlated parameters
    for col, samples in data.candidate_prior.items():
        prior_hist = get_histrogram(data.prior[col], data.bounds[col], nbins)
        posterior_hist = get_histrogram(data.posterior[col], data.bounds[col], nbins)
        candidate_prior_hist = get_histrogram(samples, data.bounds[col], nbins)
        binwidth = (data.bounds[col].high - data.bounds[col].low) / nbins

        ratio = np.where(prior_hist > ztol, np.exp(np.log(candidate_prior_hist) - np.log(prior_hist)), 0.0)
        rv = rv_histogram((ratio, np.linspace(data.bounds[col].low, data.bounds[col].high, nbins + 1)))
        weights *= rv.pdf(data.prior[col])

        bf *= (
            np.sum(
                np.where(
                    prior_hist > ztol,
                    np.exp(np.log(candidate_prior_hist) + np.log(posterior_hist) - np.log(prior_hist)),
                    0.0,
                )
            )
            * binwidth
        )

    correlated_prior = data.prior.sample(n=len(data.prior), weights=weights, replace=True, random_state=random_state)
    # Here we loop over the potentially correlated parameters
    # to compute the Bayes factor.
    for col in data.correlated_params:
        prior_hist = get_histrogram(data.prior[col], data.bounds[col], nbins)
        posterior_hist = get_histrogram(data.posterior[col], data.bounds[col], nbins)
        candidate_prior_hist = get_histrogram(correlated_prior[col], data.bounds[col], nbins)
        binwidth = (data.bounds[col].high - data.bounds[col].low) / nbins

        bf *= (
            np.sum(
                np.where(
                    prior_hist > ztol,
                    np.exp(np.log(candidate_prior_hist) + np.log(posterior_hist) - np.log(prior_hist)),
                    0.0,
                )
            )
            * binwidth
        )

    return bf
