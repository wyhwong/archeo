from typing import Callable, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

import archeo.logger
from archeo.schema import Labels
from archeo.visualization.base import clear_default_labels, initialize_plot


local_logger = archeo.logger.get_logger(__name__)


class PriorReweighter:
    """A class contains useful methods to reweight posterior samples"""

    def __init__(self, nbins: int = 100, atol: float = 1e-6) -> None:
        """Initialize the PriorReweighter

        Args:
            nbins (int): The number of bins (resolution) for numerical integration
                         or histrogram plotting.
            atol (float): The absolute tolerance
                          for numerical integration of probability distribution.
        """

        self._nbins = nbins
        self._atol = atol

        self._lb: Optional[float] = None
        self._ub: Optional[float] = None
        self._binwidth: Optional[float] = None

        self.cumulative_bayes_factor: float = 1.0

    def _compute_histogram(self, samples: pd.Series) -> np.ndarray:
        """Compute the histogram of the samples.

        The histogram is used to compute the Bayes factor between two priors.
        Check PriorReweighter.get_bayes_factor for more details.

        Args:
            samples (pd.Series): The posterior/prior samples of the parameter.

        Returns:
            hist (np.ndarray): The histogram of the samples.
        """

        hist, _ = np.histogram(
            samples,
            bins=self._nbins,
            density=True,
            range=(self._lb, self._ub),
        )
        auc = np.sum(hist) * self._binwidth

        if not np.isclose(auc, 1.0, atol=self._atol):
            msg = f"Invalid probability distribution (AUC={auc:.2f})."
            local_logger.error(msg)
            raise ValueError(msg)

        return hist

    def _setup_bounds(
        self,
        new_prior: pd.Series,
        prior: pd.Series,
        posterior: pd.Series,
    ) -> None:
        """Setup the bounds for the histograms computation.
        Suppose the bounds are ONLY valid for one Bayes factor computation.
        After the computation, the bounds should be reset.

        Args:
            new_prior (pd.Series): The new prior samples of the parameter.
            prior (pd.Series): The prior samples of the parameter.
            posterior (pd.Series): The posterior samples of the parameter.
        """

        if self._lb or self._ub:
            msg = "Bounds have already been initialized."
            local_logger.error(msg)
            raise ValueError(msg)

        self._lb: Optional[float] = min(prior.min(), posterior.min(), new_prior.min())
        self._ub: Optional[float] = max(prior.max(), posterior.max(), new_prior.max())
        self._binwidth = (self._ub - self._lb) / self._nbins

        local_logger.info("Bounds: [%.2f, %.2f], Binwidth: %.2f", self._lb, self._ub, self._binwidth)

    def _reset_bounds(self):
        """Reset the bounds for the histograms computation."""

        self._lb: Optional[float] = None
        self._ub: Optional[float] = None
        self._binwidth: Optional[float] = None

        local_logger.info("Bounds have been reset.")

    def get_bayes_factor(self, new_prior: pd.Series, prior: pd.Series, posterior: pd.Series) -> float:
        """Compute the Bayes factor between two models

        The histogram is used to compute the Bayes factor between two priors.
        NOTE: In this implementation, the likelihood function remains untouched.
        So that the Bayes factor is computed as the ratio of the new prior to the old prior.
        Details please check importance sampling.
        """

        self._setup_bounds(new_prior, prior, posterior)

        new_prior_hist = self._compute_histogram(new_prior)
        prior_hist = self._compute_histogram(prior)
        posterior_hist = self._compute_histogram(posterior)

        # Original implementation: precision not enough
        # bayes_factor = (new_prior_hist * posterior_hist / prior_hist).sum() * dtheta

        # New implementation: we take log first to avoid numerical precision issues
        # NOTE: We need to remove the 0s from the histograms to avoid log(0) = -inf
        mask = (new_prior_hist != 0) & (prior_hist != 0) & (posterior_hist != 0)
        local_logger.info("Number of bins with non-zero values: %d / %d.", mask.sum(), len(mask))

        bayes_factor = (
            np.sum(np.exp(np.log(new_prior_hist[mask]) + np.log(posterior_hist[mask]) - np.log(prior_hist[mask])))
            * self._binwidth
        )
        self.cumulative_bayes_factor *= bayes_factor

        self._reset_bounds()

        return bayes_factor

    def get_weights_func(
        self,
        samples: pd.Series,
        show: bool = True,
        param_name: Optional[str] = None,
        n_samples: Optional[int] = None,
    ) -> Callable:
        """Get the weights function using the KDE method.
        NOTE: This weights function is used for importance sampling.

        Args:
            samples (pd.Series): The prior/posterior samples of the parameter.
            show (bool): Whether to show the histogram and KDE plot.
            param_name (Optional[str]): The name of the parameter (for visualization only).
            n_samples (Optional[int]): The number of samples used to construct the KDE.

        Returns:
            weights_func (scipy.stats.gaussian_kde): The weights function.
        """

        if n_samples:
            weights_func = gaussian_kde(samples.sample(n_samples))
        else:
            weights_func = gaussian_kde(samples)

        if show:
            xlabel = f"Parameter $\\theta_i$ ({param_name})" if param_name else "Parameter $\\theta_i$"
            title = f"Distribution of {param_name}" if param_name else "Distribution of Parameter $\\theta_i$"

            _, ax = initialize_plot(labels=Labels(xlabel=xlabel, ylabel="Density", title=title))

            x = np.linspace(samples.min(), samples.max(), self._nbins)
            y = weights_func(x)
            binwidth = (samples.max() - samples.min()) / self._nbins

            sns.histplot(
                samples,
                ax=ax,
                label="Data",
                stat="density",
                binwidth=binwidth,
                fill=False,
                element="step",
            )
            sns.lineplot(y=y, x=x, label="KDE", ax=ax)

            clear_default_labels(ax)

        return weights_func
