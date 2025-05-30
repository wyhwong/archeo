from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from archeo.constants import Columns as C
from archeo.constants import Suffixes as S
from archeo.schema import Labels
from archeo.visualization import base


# Color map for the scatter plot
WHITE_VIRIDIS = LinearSegmentedColormap.from_list(
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


def distribution_summary(
    df: pd.DataFrame,
    filename="distribution_summary",
    output_dir: Optional[str] = None,
    close: bool = True,
    fmt: str = "png",
):
    """Plot the distribution of the black hole parameters.

    Args:
        df (pd.DataFrame): The prior/posterior dataframe.
        filename (str): Filename of the figure.
        output_dir (Optional[str]): Output directory.
        close (bool): Whether to close the figure.
        fmt (str): The format of the visualizations. Defaults to "png".

    Returns:
        fig (plt.Figure): Figure.
        axes (plt.Axes): Axes.
    """

    labels = Labels("Distribution of black-hole parameters")
    col_to_labels = {
        C.MASS_RATIO: "Parent Mass Ratio $q$",
        S.FINAL(C.MASS): r"Remnant Mass $m_f$ [$M_{\odot}$]",
        C.KICK: "Recoil Kick $v_f$ [$kms^{-1}$]",
        S.FINAL(C.SPIN_MAG): "Spin $\\chi_f$",
    }
    fig, axes = base.initialize_plot(nrows=4, ncols=1, figsize=(6, 8), labels=labels)
    for idx, (col, xlabel) in enumerate(col_to_labels.items()):
        sns.histplot(df[col], ax=axes[idx], element="step", fill=False, stat="density")
        axes[idx].set(xlabel=xlabel, ylabel="PDF")

    base.savefig_and_close(filename, output_dir, close, fmt)
    return (fig, axes)


def kick_against_spin_cmap(
    df: pd.DataFrame,
    filename="kick_against_spin_cmap",
    output_dir: Optional[str] = None,
    close: bool = True,
    fmt: str = "png",
):
    """Plot the remnant kick against remnant spin in cmap.

    Args:
        df (pd.DataFrame): The prior/posterior dataframe.
        filename (str): Filename of the figure.
        output_dir (Optional[str]): Output directory.
        close (bool): Whether to close the figure.
        fmt (str): The format of the visualizations. Defaults to "png".

    Returns:
        fig (plt.Figure): Figure.
        axes (plt.Axes): Axes.
    """

    # mpl_scatter_density import is for ax.scatter_density
    # implicitly used in kick_against_spin
    # pylint: disable-next=unused-import, import-outside-toplevel
    import mpl_scatter_density  # type: ignore

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
    ax.scatter_density(df[S.FINAL(C.SPIN_MAG)], df[C.KICK], cmap=WHITE_VIRIDIS)  # type: ignore
    ax.set(
        title="Remnant Kick against Remnant Spin",
        xlabel="Remnant Spin $\\chi_f$",
        ylabel=r"Remnant Kick $v_f$ [$kms^{-1}$]",
    )
    base.savefig_and_close(filename, output_dir, close, fmt)
    return (fig, ax)


def kick_distribution_on_spin(
    df: pd.DataFrame,
    filename="kick_on_spin",
    output_dir: Optional[str] = None,
    close: bool = True,
    fmt: str = "png",
):
    """Plot the distribution of remnant kick on different spin range.

    Args:
        df (pd.DataFrame): The prior/posterior dataframe.
        filename (str): Filename of the figure.
        output_dir (Optional[str]): Output directory.
        close (bool): Whether to close the figure.
        fmt (str): The format of the visualizations. Defaults to "png".

    Returns:
        fig (plt.Figure): Figure.
        axes (plt.Axes): Axes.
    """

    labels = Labels(
        title="Remnant Kick Distribution on Different Spin Range",
        xlabel="Remnant Kick $v_f$ [$km/s$]",
        ylabel="PDF",
    )
    fig, ax = base.initialize_plot(figsize=(9, 4), labels=labels)
    bounds = zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10))

    for low, high in bounds:
        data = df.loc[(low < df[S.FINAL(C.SPIN_MAG)]) & (df[S.FINAL(C.SPIN_MAG)] < high)][C.KICK]
        # To avoid extreme density values
        if len(data.index) > 100:
            density, bins = np.histogram(a=data, bins=70, density=True)
            label = rf"$\chi_f$ $\in$ $[{low:.2f}, {high:.2f}]$"
            ax.stairs(density, bins, label=label)

    plt.legend()
    base.clear_default_labels(ax)
    base.savefig_and_close(filename, output_dir, close, fmt)
    return (fig, ax)
