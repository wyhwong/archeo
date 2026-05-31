import numpy as np
import pandas as pd
from scipy.stats import rv_histogram

from archeo.bayesian.importance_sampling.resampler.base import ImportanceSamplingDataBase
from archeo.data_structures.math import Domain
from archeo.utils.logger import get_logger


LOGGER = get_logger(__name__)


def get_histogram_1d(samples: pd.Series, nbins: int, bounds: Domain) -> np.ndarray:
    """Estimate a 1D density histogram over fixed bounds.

    Args:
        samples (pd.Series): Input samples.
        nbins (int): Number of bins.
        bounds (Domain): Histogram range.

    Returns:
        np.ndarray: Density histogram values.
    """

    hist, edges = np.histogram(samples, bins=nbins, density=True, range=bounds.to_tuple())
    binwidth = edges[1] - edges[0]

    auc = np.sum(hist) * binwidth
    if not np.isclose(auc, 1.0, atol=1e-6):
        msg = f"Invalid probability distribution (AUC={auc:.2f})."
        LOGGER.warning(msg)

    return hist


class ISDataAssumeIndependence(ImportanceSamplingDataBase):
    """Importance sampling data for assume independence resampler"""

    def _get_hist_1d(self, samples: pd.Series) -> np.ndarray:
        """Compute a 1D histogram for one sample column.

        Args:
            samples (pd.Series): Sample column.

        Returns:
            np.ndarray: Density histogram values.
        """

        nbins = self.get_nbins(samples.name)
        return get_histogram_1d(samples, nbins=nbins, bounds=self.bounds[samples.name])

    def get_likelihood_samples_1d(self, random_state=42) -> np.ndarray:
        """Resample posterior to approximate likelihood under independence assumption.

        Args:
            random_state (int): Random seed for weighted sampling.

        Returns:
            np.ndarray: Resampled posterior-like samples.
        """

        weights = np.ones(len(self.posterior_samples))

        for col in self.common_columns:
            prior_hist = self._get_hist_1d(self.prior_samples[col])
            rv = rv_histogram(
                (
                    self._safe_divide(1.0, prior_hist),
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

    def _get_posterior_sample_weights_1d(self, col_name: str) -> np.ndarray:
        """Compute per-sample weights for a single column.

        Args:
            col_name (str): Column name.

        Returns:
            np.ndarray: Importance weights.
        """

        weights = np.ones(len(self.posterior_samples))

        prior_hist = self._get_hist_1d(self.prior_samples[col_name])
        new_prior_hist = self._get_hist_1d(self.new_prior_samples[col_name])
        # Avoid division by zero
        ratio = self._safe_divide(new_prior_hist, prior_hist)
        rv = rv_histogram((ratio, self.get_edges(col_name)))
        weights *= rv.pdf(self.posterior_samples[col_name])

        return weights

    def get_bayes_factor_1d(self, bootstrapping: bool = False) -> float:
        """Compute Bayes factor using factorized 1D histograms.

        Args:
            bootstrapping (bool): If ``True``, bootstrap all sample sets first.

        Returns:
            float: Bayes-factor estimate.
        """

        if bootstrapping:
            return self._get_bayes_factor_1d(
                prior_samples=self.prior_samples.sample(n=len(self.prior_samples), replace=True),
                posterior_samples=self.posterior_samples.sample(n=len(self.posterior_samples), replace=True),
                new_prior_samples=self.new_prior_samples.sample(n=len(self.new_prior_samples), replace=True),
            )

        return self._get_bayes_factor_1d(
            prior_samples=self.prior_samples,
            posterior_samples=self.posterior_samples,
            new_prior_samples=self.new_prior_samples,
        )

    def _get_bayes_factor_1d(
        self,
        prior_samples: pd.DataFrame,
        posterior_samples: pd.DataFrame,
        new_prior_samples: pd.DataFrame,
    ) -> float:
        """Compute Bayes factor from explicit dataframe inputs (1D factorized form).

        Args:
            prior_samples (pd.DataFrame): Baseline prior samples.
            posterior_samples (pd.DataFrame): Posterior samples.
            new_prior_samples (pd.DataFrame): Candidate prior samples.

        Returns:
            float: Bayes-factor estimate.
        """

        # NOTE: In this implementation, the likelihood function remains untouched.
        # So the Bayes factor is computed as the ratio of the new prior to the old prior.
        # Details please check importance sampling.

        bf = 1.0

        for col in self.common_columns:
            prior_hist = self._get_hist_1d(prior_samples[col])
            posterior_hist = self._get_hist_1d(posterior_samples[col])
            new_prior_hist = self._get_hist_1d(new_prior_samples[col])
            bf *= np.sum(posterior_hist * self._safe_divide(new_prior_hist, prior_hist)) * self.get_binwidth(col)

        return bf

    def get_reweighted_samples_1d(self, random_state=42) -> pd.DataFrame:
        """Draw posterior samples reweighted toward the candidate prior.

        Args:
            random_state (int): Random seed for weighted resampling.

        Returns:
            pd.DataFrame: Reweighted posterior samples.
        """

        weights = np.ones(len(self.posterior_samples))

        for col in self.common_columns:
            weights *= self._get_posterior_sample_weights_1d(col_name=col)

        reweighted_samples = self.posterior_samples.sample(
            n=len(self.posterior_samples),
            weights=weights,
            replace=True,
            random_state=random_state,
        )

        return reweighted_samples
