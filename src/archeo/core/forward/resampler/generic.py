import logging

import numpy as np
import pandas as pd

from archeo.core.forward.resampler.base import ImportanceSamplingDataBase
from archeo.schema import Domain


local_logger = logging.getLogger(__name__)


def get_histogram_dd(samples: np.ndarray, nbins: list[int], bounds: list[Domain]) -> np.ndarray:
    """Compute the histogram of the samples"""

    hist, edges = np.histogramdd(samples, bins=nbins, density=True, range=[b.to_tuple() for b in bounds])

    # Compute the bin volume
    binwidths = [edges[i][1] - edges[i][0] for i in range(len(edges))]
    bin_volume = np.prod(binwidths)

    auc = np.sum(hist) * bin_volume
    if not np.isclose(auc, 1.0, atol=1e-6):
        msg = f"Invalid probability distribution (AUC={auc:.2f})."
        local_logger.warning(msg)

    return hist


class ISDataGeneric(ImportanceSamplingDataBase):
    """Importance sampling data for generic resampler"""

    def _get_hist_dd(self, df_samples: pd.DataFrame) -> np.ndarray:
        """Get the histogram for a given column name"""

        nbins = [self.get_nbins(col) for col in df_samples.columns]
        bounds = [self.bounds[col] for col in df_samples.columns]
        samples_array = df_samples.to_numpy()
        return get_histogram_dd(samples_array, nbins=nbins, bounds=bounds)

    def get_likelihood_samples_dd(self, random_state=42, ztol=1e-8) -> np.ndarray:
        """Get samples for likelihood function"""

        edges = {}
        for col in self.common_columns:
            edges[col] = self.get_edges(col)

        prior_hist = self._get_hist_dd(self.prior_samples[self.common_columns])
        weights_matrix = self._safe_divide(1.0, prior_hist, ztol=ztol)

        def _get_pdf(row: pd.Series):
            idx = tuple(np.searchsorted(edges[col], row[col], side="right") - 1 for col in self.common_columns)
            return weights_matrix[idx]

        weights = self.posterior_samples.apply(_get_pdf, axis=1)

        return self.posterior_samples.sample(
            n=len(self.posterior_samples),
            weights=weights,
            replace=True,
            random_state=random_state,
        )

    def _get_posterior_sample_weights_dd(self, ztol=1e-8) -> pd.Series:
        """Get the weights for the importance sampling"""

        edges = {}
        for col in self.common_columns:
            edges[col] = self.get_edges(col)

        prior_hist = self._get_hist_dd(self.prior_samples[self.common_columns])
        new_prior_hist = self._get_hist_dd(self.new_prior_samples[self.common_columns])
        # Avoid division by zero
        weights_matrix = self._safe_divide(new_prior_hist, prior_hist, ztol=ztol)

        def _get_pdf(row: pd.Series):
            idx = tuple(np.searchsorted(edges[col], row[col], side="right") - 1 for col in self.common_columns)
            return weights_matrix[idx]

        return self.posterior_samples.apply(_get_pdf, axis=1)

    def get_bayes_factor_dd(self, bootstrapping: bool = False, ztol=1e-8) -> float:
        """Compute the Bayes factor between two models"""

        if bootstrapping:
            return self._get_bayes_factor_dd(
                prior_samples=self.prior_samples.sample(n=len(self.prior_samples), replace=True),
                posterior_samples=self.posterior_samples.sample(n=len(self.posterior_samples), replace=True),
                new_prior_samples=self.new_prior_samples.sample(n=len(self.new_prior_samples), replace=True),
                ztol=ztol,
            )

        return self._get_bayes_factor_dd(
            prior_samples=self.prior_samples,
            posterior_samples=self.posterior_samples,
            new_prior_samples=self.new_prior_samples,
            ztol=ztol,
        )

    def _get_bayes_factor_dd(
        self,
        prior_samples: pd.DataFrame,
        posterior_samples: pd.DataFrame,
        new_prior_samples: pd.DataFrame,
        ztol=1e-8,
    ) -> float:
        """Compute the Bayes factor between two models

        NOTE: In this implementation, the likelihood function remains untouched.
        So that the Bayes factor is computed as the ratio of the new prior to the old prior.
        Details please check importance sampling.
        """

        bf = 1.0

        bin_auc = np.prod([self.get_binwidth(c) for c in self.common_columns])
        new_prior_hist_bh = self._get_hist_dd(new_prior_samples[[c for c in self.common_columns]])
        prior_hist_bh = self._get_hist_dd(prior_samples[[c for c in self.common_columns]])
        posterior_hist_bh = self._get_hist_dd(posterior_samples[[c for c in self.common_columns]])
        bf *= np.sum(posterior_hist_bh * self._safe_divide(new_prior_hist_bh, prior_hist_bh, ztol=ztol)) * bin_auc

        return bf

    def get_reweighted_samples_dd(self, ztol=1e-8, random_state=42) -> pd.DataFrame:
        """Get the reweighted samples for the importance sampling"""

        weights = self._get_posterior_sample_weights_dd(ztol=ztol)

        return self.posterior_samples.sample(
            n=len(self.posterior_samples),
            weights=weights,
            replace=True,
            random_state=random_state,
        )
