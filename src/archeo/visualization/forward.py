import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interpolate

from archeo.core.forward.bayes import ISData, get_bayes_factor
from archeo.utils.helper import pre_release
from archeo.visualization.base import add_escape_velocity


@pre_release
def plot_bayes_factor_over_kick(
    ax,
    prior: pd.DataFrame,
    posterior: pd.DataFrame,
    candidate_prior: dict[str, pd.Series],
    kicks: pd.Series,
    label: str,
    n_bounds: int = 100,
    min_nk: int = 1000,
    show_escape_velocity: bool = False,
    show_scatter: bool = False,
):
    """Plot the Bayes factor over the kick range.

    Args:
        ax: The axis to plot on.
        prior: The prior DataFrame.
        posterior: The posterior DataFrame.
        candidate_prior: The candidate prior DataFrame.
        kicks: The kick values.
        label: The label of the plot.
        n_bounds: The number of bounds to interpolate.
        min_nk: The minimum number of kicks to consider.
        show_escape_velocity: Whether to show the escape velocity.
        show_scatter: Whether to show the scatter plot.

    Returns:
        The axis with the plot.
    """

    k_lb = kicks.quantile(min_nk / len(kicks))
    k_ub = kicks.max()

    _ks = np.logspace(np.log10(k_lb), np.log10(k_ub), n_bounds)
    bfs: list[float] = []

    for k in _ks:
        # Mark index of the samples that are below the kick threshold
        idx = kicks < k
        _candidate_prior = {col: ds[idx] for col, ds in candidate_prior.items()}
        data = ISData(candidate_prior=_candidate_prior, posterior=posterior, prior=prior)
        bfs.append(get_bayes_factor(data))

    f = interpolate.interp1d(_ks, bfs, fill_value="extrapolate")
    x = np.linspace(k_lb, k_ub, 1000)
    y = f(x)

    sns.lineplot(y=y, x=x, ax=ax, label=label)
    if show_scatter:
        sns.scatterplot(y=bfs, x=_ks, ax=ax, color="black")

    if show_escape_velocity:
        add_escape_velocity(ax, k_lb, max(bfs))

    return ax
