from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import archeo.logger
from archeo.constants import EscapeVelocity
from archeo.schema import Labels, Padding
from archeo.utils import file


local_logger = archeo.logger.get_logger(__name__)


def initialize_plot(
    nrows=1,
    ncols=1,
    figsize=(10, 6),
    labels=Labels(),
    padding=Padding(),
    fontsize: int = 12,
    **kwargs,
):
    """Initialize a plot from matplotlib.

    Args:
        nrows (int): The number of rows of the plot.
        ncols (int): The number of columns of the plot.
        figsize (tuple): The size of the plot.
        labels (Labels): The labels of the plot.
        padding (Padding): The padding of the plot.
        fontsize (int): The fontsize of the plot.
        **kwargs: Additional arguments for the plot.

    Returns:
        fig (matplotlib.figure.Figure): The figure of the plot.
        axes (numpy.ndarray): The axes of the plot.
    """

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    fig.tight_layout(pad=padding.tpad)
    fig.subplots_adjust(left=padding.lpad, bottom=padding.bpad)
    fig.suptitle(labels.title, fontsize=fontsize)
    fig.text(
        x=0.04,
        y=0.5,
        s=labels.ylabel,
        fontsize=fontsize,
        rotation="vertical",
        verticalalignment="center",
    )
    fig.text(x=0.5, y=0.04, s=labels.xlabel, fontsize=fontsize, horizontalalignment="center")

    if isinstance(axes, plt.Axes):
        axes.grid()
        return (fig, axes)

    for ax in axes.flat:
        ax.grid()
    return (fig, axes)


def savefig_and_close(
    filename: str,
    output_dir: Optional[str] = None,
    close: bool = True,
    fmt: str = "png",
) -> None:
    """Save the figure and close it.

    Args:
        filename (str): The filename of the figure.
        output_dir (Optional[str]): The output directory of the figure.
        close (bool): Whether to close the figure.
        fmt (str): The format of the figure.
    """

    if output_dir:
        file.check_and_create_dir(output_dir)
        savepath = f"{output_dir}/{filename}.{fmt}"
        plt.savefig(savepath, bbox_inches="tight", facecolor="w")
        local_logger.info("Saved figure to %s.", savepath)

    if close:
        plt.close()


def clear_default_labels(ax) -> None:
    """Clear the default labels of the axes.

    Args:
        ax (matplotlib.axes.Axes): The axes of the plot.
    """

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")


def add_escape_velocity(ax, v_max: float, y_max: float, log_xscale: bool = False) -> None:
    """Add escape velocity to the plot.

    Args:
        ax (plt.Axes): Axes.
        v_max (float): Maximum escape velocity.
        y_max (float): Maximum y-axis value.
        log_xscale (bool): Whether the x-axis is in log scale.
    """

    colors = iter(mcolors.TABLEAU_COLORS.keys())
    # Plot vertical lines and labels (escape velocities)
    for label, v_esc in EscapeVelocity.to_vlines().items():
        # Skip if out of scope
        if v_esc > v_max:
            return

        color = next(colors)
        ax.axvline(x=v_esc, color=color, linestyle="--", linewidth=0.5)

        text_shift = 20.0 * v_max / 3000.0
        text_shift = np.log(text_shift) if log_xscale else text_shift

        ax.text(
            v_esc + text_shift,
            0.65 * y_max,
            label,
            color=color,
            rotation=90,
            va="center",
            fontsize=12,
        )


def plot_pdf(
    ax,
    series: pd.Series,
    color: str = "blue",
    name: Optional[str] = None,
    unit: Optional[str] = None,
    ls: str = "-",
):
    """Plot the PDF of a parameter.

    Args:
        ax (plt.Axes): Axes.
        series (pd.Series): Series (pdf).
        color (str): Color.
        name (str): Name.
        unit (Optional[str]): Unit.
    """

    _series = series.dropna()
    density, bins = np.histogram(a=_series, bins=70, density=True)
    low, mid, high = (
        _series.quantile(0.05),
        _series.quantile(0.5),
        _series.quantile(0.95),
    )
    label = "%s: $%.2f_{-%.2f}^{+%.2f}$" % (
        (name or series.name),
        mid,
        mid - low,
        high - mid,
    )
    if unit:
        label += f" {unit}"
    ax.stairs(density, bins, label=label, color=color, linestyle=ls)
