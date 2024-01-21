import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import corner
from typing import Optional

import schemas.visualization
import core.visualization.base as base


def mass_estimates(
    df: pd.DataFrame,
    label: str,
    output_dir: Optional[str] = None,
    close: bool = True,
):
    """
    Plot the posterior mass estimates.

    Args:
    -----
        df (pd.DataFrame):
            The posterior dataframe.

        label (str):
            Label of the posterior.

        output_dir (Optional[str]):
            Output directory.

        close (bool):
            Whether to close the figure.

    Returns:
    -----
        fig (plt.Figure):
            Figure.

        axes (plt.Axes):
            Axes.
    """

    padding = schemas.visualization.Padding(lpad=0.13, bpad=0.14)
    labels = schemas.visualization.Labels("Distribution of Estimated Masses", "Mass [$M_{\odot}$]", "PDF")
    fig, ax = base.initialize_plot(figsize=(9, 4), labels=labels, padding=padding)

    col_to_labels = {
        "mf_": f"{label}: ",
        "m1": "Heavier Parent: ",
        "m2": "Ligher Parent: ",
    }
    for col, label_prefix in col_to_labels.items():
        density, bins = np.histogram(a=df[col], bins=70, density=True)
        inv_low, med, inv_high = (
            df[col].quantile(0.05),
            df[col].quantile(0.5),
            df[col].quantile(0.95),
        )
        ax_label = "%s: $%d_{-%d}^{+%d}$ %s" % (
            label_prefix,
            med,
            med - inv_low,
            inv_high - med,
            "[$M_{\odot}$]",
        )
        ax.stairs(density, bins, label=ax_label)
    ax.set(ylabel="", xlabel="")
    plt.legend()

    base.savefig_and_close(f"{label}_mass_estimates.png", output_dir, close)
    return (fig, ax)


def corner_estimates(
    df: pd.DataFrame,
    label: str,
    levels: list[float] = [0.68, 0.9],
    nbins: int = 70,
    output_dir: Optional[str] = None,
    close: bool = True,
):
    """
    Plot the posterior corner plot.

    Args:
    -----
        df (pd.DataFrame):
            The posterior dataframe.

        label (str):
            Label of the posterior.

        levels (list[float]):
            Contour levels.

        nbins (int):
            Number of bins.

        output_dir (Optional[str]):
            Output directory.

        close (bool):
            Whether to close the figure.

    Returns:
    -----
        fig (plt.Figure):
            Figure.
    """

    fig = corner.corner(
        df,
        nbins,
        var_names=["m1", "m2", "mf_", "vf", "chif"],
        labels=["$m_1$", "$m_2$", "$m_f$", "$v_f$", "$\chi_f$"],
        levels=levels,
        plot_density=True,
        plot_samples=False,
        color="blue",
        fill_contours=False,
        smooth=True,
        plot_datapoints=False,
        hist_kwargs=dict(density=True),
    )
    plt.legend()
    base.savefig_and_close(f"{label}_corner.png", output_dir, close)
    return fig


def cumulative_kick_probability_curve(
    df: pd.DataFrame,
    label: str,
    output_dir: Optional[str] = None,
    close: bool = True,
):
    """
    Plot the cumulative kick probability curve.

    Args:
    -----
        df (pd.DataFrame):
            The posterior dataframe.

        label (str):
            Label of the posterior.

        output_dir (Optional[str]):
            Output directory.

        close (bool):
            Whether to close the figure.

    Returns:
    -----
        fig (plt.Figure):
            Figure.

        axes (plt.Axes):
            Axes.
    """

    padding = schemas.visualization.Padding(bpad=0.14)
    labels = schemas.visualization.Labels("Cumulative Kick Probability Curve", "Recoil Velocity $v_f$ ($km/s$)", "CDF")
    fig, ax = base.initialize_plot(figsize=(9, 4), labels=labels, padding=padding)

    df_under_PISN = df.loc[(df["m1"] < 65) & (df["m2"] < 65)]
    norm_factor = len(df_under_PISN) / len(df)
    density, bins = np.histogram(df_under_PISN["vf"], 70, density=True)
    dbin = bins[1] - bins[0]
    density *= norm_factor
    cdf = np.cumsum(density) * dbin
    x = bins[:-1] + dbin / 2
    sns.lineplot(y=[0] + list(cdf), x=[0] + list(x), ax=ax, label=label)
    ax.set(ylabel="", xlabel="")
    plt.legend()

    base.savefig_and_close(f"{label}_cumulative_kick_probability_curve.png", output_dir, close)
    return (fig, ax)
