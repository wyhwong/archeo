import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

import schemas.visualization
import core.visualization.base as base


def distribution(
    df: pd.DataFrame,
    filename="prior_distribution.png",
    output_dir: Optional[str] = None,
    close: bool = True,
):
    """
    Plot the distribution of the prior parameters.

    Args:
    -----
        df (pd.DataFrame):
            The prior dataframe.

        filename (str):
            Filename of the figure.

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

    labels = schemas.visualization.Labels("Distribution of remnant black-hole parameters")
    col_to_labels = {
        "q": {"x": "Parent Mass Ratio $q$", "y": "PDF"},
        "mf": {"x": "Remnant Mass $m_f$ [$M_{\odot}$]", "y": "PDF"},
        "vf": {"x": "Recoil Kick $v_f$ [$kms^{-1}$]", "y": "PDF"},
        "chif": {"x": "Spin $\\chi_f$", "y": "PDF"},
    }
    fig, axes = base.initialize_plot(nrows=4, ncols=1, figsize=(6, 8), labels=labels)
    for index, (col, labels) in enumerate(col_to_labels.items()):
        sns.histplot(
            df[col],
            ax=axes[index],
            element="step",
            fill=False,
            stat="density",
        )
        axes[index].set(xlabel=labels["x"], ylabel=labels["y"])

    base.savefig_and_close(filename, output_dir, close)
    return (fig, axes)


def kick_against_spin(
    df: pd.DataFrame,
    filename="prior_kick_against_spin.png",
    output_dir: Optional[str] = None,
    close: bool = True,
):
    """
    Plot the distribution of the prior parameters.

    Args:
    -----
        df (pd.DataFrame):
            The prior dataframe.

        filename (str):
            Filename of the figure.

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

    padding = schemas.visualization.Padding(lpad=0.13)
    labels = schemas.visualization.Labels(
        "Remnant Kick against Remnant Spin",
        "Remnant Spin $\\chi_f$",
        "Remnant Kick $v_f$ [$kms^{-1}$]",
    )
    fig, ax = base.initialize_plot(figsize=(8, 6), labels=labels, padding=padding)

    # Plot the scatter plot
    sns.scatterplot(data=df, x="chif", y="vf", s=5, color=".15", ax=ax)
    ax.set(xlabel="", ylabel="")

    # Save once
    base.savefig_and_close(f"scatter_{filename}", output_dir, close=False)

    # Plot the contour plot
    sns.histplot(data=df, x="chif", y="vf", bins=120, pthresh=0.05, ax=ax)
    sns.kdeplot(data=df, x="chif", y="vf", levels=[0.1, 0.3, 0.5], color="b", linewidths=1, ax=ax)
    ax.set(xlabel="", ylabel="")

    base.savefig_and_close(filename, output_dir, close)
    return (fig, ax)


def kick_distribution_on_spin(
    df: pd.DataFrame,
    filename="prior_kick_on_spin.png",
    output_dir: Optional[str] = None,
    close: bool = True,
):
    """
    Plot the distribution of the prior parameters.

    Args:
    -----
        df (pd.DataFrame):
            The prior dataframe.

        filename (str):
            Filename of the figure.

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

    labels = schemas.visualization.Labels(
        title="Remnant Kick Distribution on Different Spin Range",
        xlabel="Remnant Kick $v_f$ [$km/s$]",
        ylabel="PDF",
    )
    fig, ax = base.initialize_plot(figsize=(9, 4), labels=labels)
    bounds = zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10))

    for low_bound, up_bound in bounds:
        data = df.loc[(low_bound < df["chif"]) & (df["chif"] < up_bound)]["vf"]
        # To avoid extreme density values
        if len(data.index) > 100:
            density, bins = np.histogram(a=data, bins=70, density=True)
            ax.stairs(
                density,
                bins,
                label=f"$\chi_f$ $\in$ $[{low_bound:.2f}, {up_bound:.2f}]$",
            )
    ax.set(xlabel="", ylabel="")
    plt.legend()

    base.savefig_and_close(filename, output_dir, close)
    return (fig, ax)
