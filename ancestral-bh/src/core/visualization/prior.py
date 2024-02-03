from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

import core.visualization.base as base
import schemas.visualization


white_viridis = LinearSegmentedColormap.from_list(
    "white_viridis",
    [  # type: ignore
        (0, "#ffffff"),
        (1e-20, "#440053"),
        (0.2, "#404388"),
        (0.4, "#2a788e"),
        (0.6, "#21a784"),
        (0.8, "#78d151"),
        (1, "#fde624"),
    ],
    N=256,
)


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
        "mf": {"x": r"Remnant Mass $m_f$ [$M_{\odot}$]", "y": "PDF"},
        "vf": {"x": "Recoil Kick $v_f$ [$kms^{-1}$]", "y": "PDF"},
        "chif": {"x": "Spin $\\chi_f$", "y": "PDF"},
    }
    fig, axes = base.initialize_plot(nrows=4, ncols=1, figsize=(6, 8), labels=labels)
    for index, (col, line_labels) in enumerate(col_to_labels.items()):
        sns.histplot(
            df[col],
            ax=axes[index],
            element="step",
            fill=False,
            stat="density",
        )
        axes[index].set(xlabel=line_labels["x"], ylabel=line_labels["y"])

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

    # mpl_scatter_density import is for ax.scatter_density
    # implicitly used in kick_against_spin
    # pylint: disable-next=unused-import, import-outside-toplevel
    import mpl_scatter_density  # type: ignore

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
    ax.scatter_density(df["chif"], df["vf"], cmap=white_viridis)  # type: ignore
    ax.set(
        title="Remnant Kick against Remnant Spin",
        xlabel="Remnant Spin $\\chi_f$",
        ylabel=r"Remnant Kick $v_f$ [$kms^{-1}$]",
    )

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
                label=rf"$\chi_f$ $\in$ $[{low_bound:.2f}, {up_bound:.2f}]$",
            )
    ax.set(xlabel="", ylabel="")
    plt.legend()

    base.savefig_and_close(filename, output_dir, close)
    return (fig, ax)
