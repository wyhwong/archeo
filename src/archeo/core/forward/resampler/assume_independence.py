import logging

import numpy as np
import pandas as pd
from scipy.stats import rv_histogram

from archeo.core.forward.resampler.base import ImportanceSamplingDataBase
from archeo.schema import Domain


local_logger = logging.getLogger(__name__)


def get_histogram_1d(samples: pd.Series, nbins: int, bounds: Domain) -> np.ndarray:
    """Compute the histogram of the samples"""

    hist, edges = np.histogram(samples, bins=nbins, density=True, range=bounds.to_tuple())
    binwidth = edges[1] - edges[0]

    auc = np.sum(hist) * binwidth
    if not np.isclose(auc, 1.0, atol=1e-6):
        msg = f"Invalid probability distribution (AUC={auc:.2f})."
        local_logger.warning(msg)

    return hist


class ISDataAssumeIndependence(ImportanceSamplingDataBase):
    """Importance sampling data for assume independence resampler"""

    def _get_hist_1d(self, samples: pd.Series) -> np.ndarray:
        """Get the histogram for a given column name"""

        nbins = self.get_nbins(samples.name)
        return get_histogram_1d(samples, nbins=nbins, bounds=self.bounds[samples.name])

    def get_likelihood_samples_1d(self, random_state=42, ztol=1e-8) -> np.ndarray:
        """Get samples for likelihood function"""

        weights = np.ones(len(self.posterior_samples))

        for col in self.common_columns:
            prior_hist = self._get_hist_1d(self.prior_samples[col])
            rv = rv_histogram(
                (
                    self._safe_divide(1.0, prior_hist, ztol=ztol),
                    self.get_edges(col_name=col),
                )
            )
            weights *= rv.pdf(self.posterior_samples[col])

        return self.posterior_samples.sample(
            n=len(self.posterior_samples),
            weights=weights,
            replace=True,
            random_state=random_state,
        )

    def get_weights_1d(self, col_name: str, ztol=1e-8) -> np.ndarray:
        """Get the weights for the importance sampling"""

        weights = np.ones(len(self.posterior_samples))

        prior_hist = self._get_hist_1d(self.prior_samples[col_name])
        new_prior_hist = self._get_hist_1d(self.new_prior_samples[col_name])
        # Avoid division by zero
        ratio = self._safe_divide(new_prior_hist, prior_hist, ztol=ztol)
        rv = rv_histogram((ratio, self.get_edges(col_name)))
        weights *= rv.pdf(self.posterior_samples[col_name])

        return weights

    def get_bayes_factor_1d(self, ztol=1e-8) -> float:
        """Compute the Bayes factor between two models

        NOTE: In this implementation, the likelihood function remains untouched.
        So that the Bayes factor is computed as the ratio of the new prior to the old prior.
        Details please check importance sampling.
        """

        bf = 1.0
        weights = np.ones(len(self.prior_samples))

        for col in self.common_columns:

            prior_hist = self._get_hist_1d(self.prior_samples[col])
            posterior_hist = self._get_hist_1d(self.posterior_samples[col])
            new_prior_hist = self._get_hist_1d(self.new_prior_samples[col])

            ratios = self._safe_divide(new_prior_hist, prior_hist, ztol=ztol)
            rv = rv_histogram((ratios, self.get_edges(col_name=col)))
            weights *= rv.pdf(self.prior_samples[col])

            bf *= np.sum(new_prior_hist * self._safe_divide(posterior_hist, prior_hist, ztol=ztol)) * self.get_binwidth(
                col
            )

        # Since weights are all zero, return BF=0
        # Because the priors are non-overlapping
        if weights.sum() == 0:
            return 0.0

        return bf

    def get_reweighted_samples_1d(self, ztol=1e-8, random_state=42) -> pd.DataFrame:
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
