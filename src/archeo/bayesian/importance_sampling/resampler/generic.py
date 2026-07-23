from typing import Optional

import numpy as np
import pandas as pd

from archeo.bayesian.importance_sampling.resampler.base import ImportanceSamplingDataBase
from archeo.data_structures.math import Domain
from archeo.utils.logger import get_logger


LOGGER = get_logger(__name__)


def get_histogram_dd(samples: np.ndarray, nbins: list[int], bounds: list[Domain]) -> np.ndarray:
    """Estimate a multi-dimensional density histogram over fixed bounds.

    Args:
        samples (np.ndarray): Sample matrix of shape ``(n, d)``.
        nbins (list[int]): Bin counts per dimension.
        bounds (list[Domain]): Domain bounds per dimension.

    Returns:
        np.ndarray: Multi-dimensional density histogram.
    """

    hist, edges = np.histogramdd(samples, bins=nbins, density=True, range=[b.to_tuple() for b in bounds])

    # Compute the bin volume
    binwidths = [edges[i][1] - edges[i][0] for i in range(len(edges))]
    bin_volume = np.prod(binwidths)

    auc = np.sum(hist) * bin_volume
    if not np.isclose(auc, 1.0, atol=1e-6):
        msg = f"Invalid probability distribution (AUC={auc:.2f})."
        LOGGER.warning(msg)

    return hist


class ISDataGeneric(ImportanceSamplingDataBase):
    """Importance sampling data for generic resampler"""

    def _get_hist_dd(self, df_samples: pd.DataFrame) -> np.ndarray:
        """Compute multi-dimensional histogram from dataframe columns.

        Args:
            df_samples (pd.DataFrame): Input samples.

        Returns:
            np.ndarray: Density histogram.
        """

        nbins = [self.get_nbins(col) for col in df_samples.columns]
        bounds = [self.bounds[col] for col in df_samples.columns]
        samples_array = df_samples.to_numpy()
        return get_histogram_dd(samples_array, nbins=nbins, bounds=bounds)

    def get_likelihood_samples_dd(self, random_state: Optional[int] = None) -> np.ndarray:
        """Resample posterior to approximate likelihood in joint space.

        Args:
            random_state (int): Random seed for weighted sampling.

        Returns:
            np.ndarray: Resampled posterior-like samples.
        """

        edges = {}
        for col in self.common_columns:
            edges[col] = self.get_edges(col)

        prior_hist = self._get_hist_dd(self.prior_samples[self.common_columns])
        weights_matrix = self._safe_divide(1.0, prior_hist)

        def _get_pdf(row: pd.Series):
            indices = tuple(np.searchsorted(edges[col], row[col], side="right") - 1 for col in self.common_columns)
            indices = tuple(max(0, min(weights_matrix.shape[i] - 1, idx)) for i, idx in enumerate(indices))
            return weights_matrix[indices]

        weights = self.posterior_samples.apply(_get_pdf, axis=1)

        return self.posterior_samples.sample(
            n=len(self.posterior_samples),
            weights=weights,
            replace=True,
            random_state=random_state,
        )

    def _get_posterior_sample_weights_dd(self) -> pd.Series:
        """Compute multi-dimensional importance weights for posterior rows.

        Returns:
            pd.Series: Row-wise importance weights.
        """

        edges = {}
        for col in self.common_columns:
            edges[col] = self.get_edges(col)

        prior_hist = self._get_hist_dd(self.prior_samples[self.common_columns])
        new_prior_hist = self._get_hist_dd(self.new_prior_samples[self.common_columns])
        # Avoid division by zero
        weights_matrix = self._safe_divide(new_prior_hist, prior_hist)

        def _get_pdf(row: pd.Series):
            indices = tuple(np.searchsorted(edges[col], row[col], side="right") - 1 for col in self.common_columns)
            indices = tuple(max(0, min(weights_matrix.shape[i] - 1, idx)) for i, idx in enumerate(indices))
            return weights_matrix[indices]

        return self.posterior_samples.apply(_get_pdf, axis=1)

    def get_bayes_factor_dd(self, bootstrapping: bool = False) -> float:
        """Compute Bayes factor using joint multi-dimensional histograms.

        Args:
            bootstrapping (bool): If ``True``, bootstrap all sample sets first.

        Returns:
            float: Bayes-factor estimate.
        """

        if bootstrapping:
            return self._get_bayes_factor_dd(
                prior_samples=self.prior_samples.sample(n=len(self.prior_samples), replace=True),
                posterior_samples=self.posterior_samples.sample(n=len(self.posterior_samples), replace=True),
                new_prior_samples=self.new_prior_samples.sample(n=len(self.new_prior_samples), replace=True),
            )

        return self._get_bayes_factor_dd(
            prior_samples=self.prior_samples,
            posterior_samples=self.posterior_samples,
            new_prior_samples=self.new_prior_samples,
        )

    def _get_bayes_factor_dd(
        self,
        prior_samples: pd.DataFrame,
        posterior_samples: pd.DataFrame,
        new_prior_samples: pd.DataFrame,
    ) -> float:
        """Compute Bayes factor from explicit dataframe inputs (joint form).

        Args:
            prior_samples (pd.DataFrame): Baseline prior samples.
            posterior_samples (pd.DataFrame): Posterior samples.
            new_prior_samples (pd.DataFrame): Candidate prior samples.

        Returns:
            float: Bayes-factor estimate.
        """

        # NOTE: In this implementation, the likelihood function remains untouched.
        # So that the Bayes factor is computed as the ratio of the new prior to the old prior.
        # Details please check importance sampling.

        bf = 1.0

        bin_auc = np.prod([self.get_binwidth(c) for c in self.common_columns])
        new_prior_hist_bh = self._get_hist_dd(new_prior_samples[[c for c in self.common_columns]])
        prior_hist_bh = self._get_hist_dd(prior_samples[[c for c in self.common_columns]])
        posterior_hist_bh = self._get_hist_dd(posterior_samples[[c for c in self.common_columns]])
        bf *= np.sum(posterior_hist_bh * self._safe_divide(new_prior_hist_bh, prior_hist_bh)) * bin_auc

        return bf

    def get_reweighted_samples_dd(self, random_state: Optional[int] = None) -> pd.DataFrame:
        """Draw posterior samples reweighted in joint parameter space.

        Args:
            random_state (int): Random seed for weighted resampling.

        Returns:
            pd.DataFrame: Reweighted posterior samples.
        """

        weights = self._get_posterior_sample_weights_dd()

        return self.posterior_samples.sample(
            n=len(self.posterior_samples),
            weights=weights,
            replace=True,
            random_state=random_state,
        )
