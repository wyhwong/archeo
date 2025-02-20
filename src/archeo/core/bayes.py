from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interpolate
from scipy.stats import gaussian_kde
from tqdm import tqdm

import archeo.logger
from archeo.constants import Columns as C
from archeo.schema import Labels
from archeo.visualization.base import add_escape_velocity, clear_default_labels, initialize_plot


local_logger = archeo.logger.get_logger(__name__)


class BayesFactorCalculator:
    """A class to compute the Bayes factor between two priors."""

    def __init__(self, nbins: int = 100, atol: float = 1e-6, use_kde: bool = False) -> None:
        """Initialize the BayesFactorCalculator

        Args:
            nbins (int): The number of bins (resolution) for numerical integration
                         or histrogram plotting.
            atol (float): The absolute tolerance
                          for numerical integration of probability distribution.
            use_kde (bool): Whether to use the KDE method to compute the weights function.
        """

        self._nbins = nbins
        self._atol = atol
        self._use_kde = use_kde

        self._lb: Optional[float] = None
        self._ub: Optional[float] = None
        self._binwidth: Optional[float] = None
        self._bincenters: Optional[np.ndarray] = None

    def _setup_bounds(self, *series_collection: pd.Series) -> None:
        """Setup the bounds for the histograms computation.
        Suppose the bounds are ONLY valid for one Bayes factor computation.
        After the computation, the bounds should be reset.

        Args:
            series_collection (pd.Series): The posterior/prior samples of the parameter.
        """

        if self._lb or self._ub:
            local_logger.warning("Bounds are initialized. Overwrote the bounds.")

        self._lb: Optional[float] = min(series.min() for series in series_collection)
        self._ub: Optional[float] = max(series.max() for series in series_collection)
        self._binwidth = (self._ub - self._lb) / self._nbins

        bin_edges = np.linspace(self._lb, self._ub, self._nbins + 1)
        self._bincenters = (bin_edges[:-1] + bin_edges[1:]) / 2

        local_logger.info("Bounds: [%.2f, %.2f], Binwidth: %.2f", self._lb, self._ub, self._binwidth)

    def _reset_bounds(self):
        """Reset the bounds for the histograms computation."""

        self._lb: Optional[float] = None
        self._ub: Optional[float] = None
        self._binwidth: Optional[float] = None
        self._bincenters: Optional[np.ndarray] = None

        local_logger.info("Bounds have been reset.")

    def get_hist(self, samples: pd.Series) -> np.ndarray:
        """Compute the histogram of the samples.

        The histogram is used to compute the Bayes factor between two priors.
        Check PriorReweighter.get_bayes_factor for more details.

        Args:
            samples (pd.Series): The posterior/prior samples of the parameter.

        Returns:
            hist (np.ndarray): The histogram of the samples.
        """

        if self._use_kde:
            weights_func = gaussian_kde(samples)
            hist = weights_func(self._bincenters)
            # Normalize the histogram to ensure the AUC = 1
            hist /= np.sum(hist) * self._binwidth

        else:
            hist, _ = np.histogram(samples, bins=self._nbins, density=True, range=(self._lb, self._ub))

        auc = np.sum(hist) * self._binwidth
        if not np.isclose(auc, 1.0, atol=self._atol):
            msg = f"Invalid probability distribution (AUC={auc:.2f})."
            local_logger.error(msg)
            raise ValueError(msg)

        return hist

    def get_bayes_factor(
        self,
        candidate_prior_param: pd.Series,
        prior_param: pd.Series,
        posterior_param: pd.Series,
    ) -> float:
        """Compute the Bayes factor between two models

        NOTE: In this implementation, the likelihood function remains untouched.
        So that the Bayes factor is computed as the ratio of the new prior to the old prior.
        Details please check importance sampling.

        Args:
            candidate_prior_param (pd.Series): The candidate prior samples of the parameter.
            prior_param (pd.Series): The prior samples of the parameter.
            posterior_param (pd.Series): The posterior samples of the parameter.

        Returns:
            bayes_factor (float): The Bayes factor between the candidate prior and the prior.
        """

        self._setup_bounds(candidate_prior_param, prior_param, posterior_param)

        new_prior_hist = self.get_hist(candidate_prior_param)
        prior_hist = self.get_hist(prior_param)
        posterior_hist = self.get_hist(posterior_param)

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

        self._reset_bounds()

        return bayes_factor

    def get_likelihood_hist(self, prior: pd.Series, posterior: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Compute the likelihood histogram of the parameter."""

        self._setup_bounds(prior, posterior)

        prior_hist = self.get_hist(prior)
        posterior_hist = self.get_hist(posterior)

        likelihood_hist = posterior_hist / prior_hist
        # Replace NaNs/infs with zeros
        likelihood_hist = np.nan_to_num(likelihood_hist, posinf=0.0, neginf=0.0)
        # normalize the likelihood
        likelihood_hist /= np.sum(likelihood_hist) * self._binwidth
        bincenters = self._bincenters

        self._reset_bounds()

        return likelihood_hist, bincenters

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

    @staticmethod
    def _get_kick_bounds(
        candidate_prior_bh1: Optional[pd.DataFrame] = None,
        candidate_prior_bh2: Optional[pd.DataFrame] = None,
        least_n_samples: int = 20000,
    ) -> tuple[float, float]:
        """Get the lower and upper bounds of the BH kick for plotting."""

        if (candidate_prior_bh1 is None) and (candidate_prior_bh2 is None):
            raise ValueError("Both candidate priors are None.")

        lb_kick_1 = (
            np.inf
            if candidate_prior_bh1 is None
            else candidate_prior_bh1[C.BH_KICK].sort_values(ascending=True).iloc[least_n_samples]
        )
        lb_kick_2 = (
            np.inf
            if candidate_prior_bh2 is None
            else candidate_prior_bh2[C.BH_KICK].sort_values(ascending=True).iloc[least_n_samples]
        )
        ub_kick_1 = -np.inf if candidate_prior_bh1 is None else candidate_prior_bh1[C.BH_KICK].max()
        ub_kick_2 = -np.inf if candidate_prior_bh2 is None else candidate_prior_bh2[C.BH_KICK].max()

        ub_kick = max(ub_kick_1, ub_kick_2)

        if lb_kick_1 == np.inf:
            lb_kick = lb_kick_2
        elif lb_kick_2 == np.inf:
            lb_kick = lb_kick_1
        else:
            lb_kick = min(lb_kick_1, lb_kick_2)

        return (lb_kick, ub_kick)

    @staticmethod
    def _plot_bayes_factor_over_kick(
        kicks: np.ndarray,
        bfs: np.ndarray,
        ax: plt.Axes,
        label: str,
    ) -> None:
        """Plot the Bayes factor over the BH kick."""

        f = interpolate.interp1d(kicks, bfs)
        x = np.linspace(kicks.min(), kicks.max(), 1000)
        y = f(x)

        sns.lineplot(y=y, x=x, ax=ax, label=label)
        sns.scatterplot(y=bfs, x=kicks, ax=ax, color="black")

    def _get_bayes_factor_from_data(
        self,
        data: dict[str, Union[pd.Series, pd.DataFrame]],
        kick: float,
    ) -> float:
        """Get the Bayes factor from the data. (See plot_bayes_factor_over_kick)"""

        bf = 1.0
        prior = data["candidate_prior"].loc[data["candidate_prior"][C.BH_KICK] <= kick]

        if data.get("mass_prior") is not None:
            bf *= self.get_bayes_factor(
                candidate_prior_param=prior[C.BH_MASS],
                prior_param=data["mass_prior"],
                posterior_param=data["mass_posterior"],
            )

        if data.get("spin_prior") is not None:
            bf *= self.get_bayes_factor(
                candidate_prior_param=prior[C.BH_SPIN],
                prior_param=data["spin_prior"],
                posterior_param=data["spin_posterior"],
            )

        return bf

    def plot_bayes_factor_over_kick(
        self,
        ax: plt.Axes,
        label: str,
        data_bh1: Optional[dict[str, Union[pd.Series, pd.DataFrame]]] = None,
        data_bh2: Optional[dict[str, Union[pd.Series, pd.DataFrame]]] = None,
        n_bounds: int = 30,
        least_n_samples: int = 20000,
        show_escape_velocity: bool = True,
    ) -> None:
        """Plot the Bayes factor over the BH kick.

        Args:
            ax (plt.Axes): The matplotlib axes for plotting.
            label (str): The label of the Bayes factor.
            data_bh1 (Optional[dict[str, Union[pd.Series, pd.DataFrame]]): The data of the first BH.
            data_bh2 (Optional[dict[str, Union[pd.Series, pd.DataFrame]]): The data of the second BH.
            n_bounds (int): The number of bounds for the BH kick.
            least_n_samples (int): The least number of samples for the BH kick.
            show_escape_velocity (bool): Whether to show the escape velocity on the plot.

        Expected data structure of data_bh1 and data_bh2:
        data = {
            "mass_prior": (Optional, pd.Series),
            "mass_posterior": (Optional, pd.Series),
            "spin_prior": (Optional, pd.Series),
            "spin_posterior": (Optional, pd.Series),
            "candidate_prior": (Optional, pd.DataFrame),
        }
        Assumptions:
        1. The candidate prior must exist if dict is not None.
        2. Only if mass and spin priors are provided, we replace the prior with the candidate prior.
        """

        lb_kick, ub_kick = self._get_kick_bounds(
            candidate_prior_bh1=data_bh1["candidate_prior"] if data_bh1 else None,
            candidate_prior_bh2=data_bh2["candidate_prior"] if data_bh2 else None,
            least_n_samples=least_n_samples,
        )

        kicks = np.logspace(np.log10(lb_kick), np.log10(ub_kick), n_bounds)
        bfs: list[float] = []

        for kick in tqdm(kicks):
            bf = self._get_bayes_factor_from_data(data_bh1, kick) if data_bh1 else 1.0
            bf *= self._get_bayes_factor_from_data(data_bh2, kick) if data_bh2 else 1.0
            bfs.append(bf)

        if show_escape_velocity:
            add_escape_velocity(ax, ub_kick, max(bfs))

        return self._plot_bayes_factor_over_kick(kicks, np.array(bfs), ax, label)
