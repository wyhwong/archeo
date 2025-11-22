from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rv_histogram

import archeo.logger
from archeo.schema import Domain
from archeo.utils.helper import pre_release


local_logger = archeo.logger.get_logger(__name__)


def get_histogram(samples: pd.Series, nbins: int, bounds: Domain) -> np.ndarray:
    """Compute the histogram of the samples"""

    hist, edges = np.histogram(samples, bins=nbins, density=True, range=(bounds.low, bounds.high))
    binwidth = edges[1] - edges[0]

    auc = np.sum(hist) * binwidth
    if not np.isclose(auc, 1.0, atol=1e-6):
        msg = f"Invalid probability distribution (AUC={auc:.2f})."
        local_logger.warning(msg)

    return hist


def _safe_divide(a: np.ndarray, b: np.ndarray, ztol: float = 1e-8) -> np.ndarray:
    """Safe division to avoid division by zero"""

    return np.where(b > ztol, np.exp(np.log(a) - np.log(b)), 0.0)


@dataclass
class ImportanceSamplingData:
    """Importance sampling data"""

    posterior_samples: pd.DataFrame
    prior_samples: pd.DataFrame
    new_prior_samples: pd.DataFrame
    binsize_spin: float = 0.05
    binsize_mass: float = 1.0

    def __post_init__(self) -> None:
        """Compute the bounds of the dataframes"""

        self._bounds = self._compute_bounds()

    @property
    def common_columns(self) -> list[str]:
        """Get the common columns between posterior and prior samples"""

        return list(
            set(self.posterior_samples.columns)
            .intersection(set(self.prior_samples.columns))
            .intersection(set(self.new_prior_samples.columns))
        )

    @property
    def bounds(self) -> dict[str, Domain]:
        """Get the bounds of the dataframes"""

        return self._bounds

    def get_binsize(self, col_name: str) -> Optional[float]:

        if col_name.startswith("a"):
            return self.binsize_spin

        if col_name.startswith("m"):
            return self.binsize_mass

        raise ValueError(f"Unknown column name {col_name}")

    def get_nbins(self, col_name: str) -> Optional[int]:
        """Get the number of bins for a given column name"""

        binsize = self.get_binsize(col_name)
        bounds = self.bounds[col_name]
        return int((bounds.high - bounds.low) / binsize)

    def _get_hist(self, samples: pd.Series) -> np.ndarray:
        """Get the histogram for a given column name"""

        nbins = self.get_nbins(samples.name)
        return get_histogram(samples, nbins=nbins, bounds=self.bounds[samples.name])

    def _compute_bounds(self) -> dict[str, Domain]:
        """Get the bounds of the dataframes"""

        bounds: dict[str, Domain] = {}

        for col in self.common_columns:
            _min = min(
                self.posterior_samples[col].min(),
                self.prior_samples[col].min(),
                self.new_prior_samples[col].min(),
            )
            _max = max(
                self.posterior_samples[col].max(),
                self.prior_samples[col].max(),
                self.new_prior_samples[col].max(),
            )
            bounds[col] = Domain(low=_min, high=_max)

        return bounds

    def get_edges(self, col_name: str) -> np.ndarray:
        """Get the edges of the histogram for a given column name"""

        nbins = self.get_nbins(col_name)
        bounds = self.bounds[col_name]
        return np.linspace(bounds.low, bounds.high, nbins + 1)

    def get_binwidth(self, col_name: str) -> float:
        """Get the bin width for a given column name"""

        edges = self.get_edges(col_name)
        return edges[1] - edges[0]

    @pre_release
    def get_likelihood_samples(self, random_state=42, ztol=1e-8) -> np.ndarray:
        """Get samples for likelihood function"""

        weights = np.ones(len(self.posterior_samples))

        for col in self.common_columns:
            prior_hist = self._get_hist(self.prior_samples[col])
            rv = rv_histogram((_safe_divide(1.0, prior_hist, ztol=ztol), self.get_edges(col_name=col)))
            weights *= rv.pdf(self.posterior_samples[col])

        return self.posterior_samples.sample(
            n=len(self.posterior_samples),
            weights=weights,
            replace=True,
            random_state=random_state,
        )

    @pre_release
    def get_weights(self, col_name: str, ztol=1e-8) -> np.ndarray:
        """Get the weights for the importance sampling"""

        weights = np.ones(len(self.posterior_samples))

        prior_hist = self._get_hist(self.prior_samples[col_name])
        new_prior_hist = self._get_hist(self.new_prior_samples[col_name])
        # Avoid division by zero
        ratio = _safe_divide(new_prior_hist, prior_hist, ztol=ztol)
        rv = rv_histogram((ratio, self.get_edges(col_name)))
        weights *= rv.pdf(self.posterior_samples[col_name])

        return weights

    @pre_release
    def get_bayes_factor(self, ztol=1e-8) -> float:
        """Compute the Bayes factor between two models

        NOTE: In this implementation, the likelihood function remains untouched.
        So that the Bayes factor is computed as the ratio of the new prior to the old prior.
        Details please check importance sampling.
        """

        bf = 1.0
        weights = np.ones(len(self.prior_samples))

        for col in self.common_columns:

            prior_hist = self._get_hist(self.prior_samples[col])
            posterior_hist = self._get_hist(self.posterior_samples[col])
            new_prior_hist = self._get_hist(self.new_prior_samples[col])

            ratios = _safe_divide(new_prior_hist, prior_hist, ztol=ztol)
            rv = rv_histogram((ratios, self.get_edges(col_name=col)))
            weights *= rv.pdf(self.prior_samples[col])

            bf *= np.sum(new_prior_hist * _safe_divide(posterior_hist, prior_hist, ztol=ztol)) * self.get_binwidth(col)

        # Since weights are all zero, return BF=0
        # Because the priors are non-overlapping
        if weights.sum() == 0:
            return 0.0

        # # Handle the correlation between parameters
        # inferred_new_prior_samples = self.prior_samples.sample(
        #     n=len(self.prior_samples),
        #     weights=weights,
        #     replace=True,
        #     random_state=random_state,
        # )
        # for col in self.posterior_samples:
        #     if col in self.new_prior_samples.columns:
        #         continue
        #     if col not in self.prior_samples.columns:
        #         continue
        #     prior_hist = self._get_hist(self.prior_samples[col])
        #     posterior_hist = self._get_hist(self.posterior_samples[col])
        #     new_prior_hist = self._get_hist(inferred_new_prior_samples[col])
        #     bf *= np.sum(
        #         new_prior_hist * _safe_divide(posterior_hist, prior_hist, ztol=ztol)
        #     ) * self.get_binwidth(col)

        return bf

    @pre_release
    def get_reweighted_samples(self, ztol=1e-8, random_state=42) -> pd.DataFrame:
        """Get the reweighted samples for the importance sampling"""

        weights = np.ones(len(self.posterior_samples))

        for col in self.common_columns:
            weights *= self.get_weights(col_name=col, ztol=ztol)

        reweighted_samples = self.posterior_samples.sample(
            n=len(self.posterior_samples),
            weights=weights,
            replace=True,
            random_state=random_state,
        )

        return reweighted_samples


# Alias
ISData = ImportanceSamplingData
