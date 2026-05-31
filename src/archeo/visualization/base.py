from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from archeo.constants.physics import TypicalHostEscapeVelocity
from archeo.data_structures.visualization import Labels, Padding
from archeo.utils.fs import check_and_create_dir
from archeo.utils.logger import get_logger


LOGGER = get_logger(__name__)


def initialize_plot(
    nrows=1,
    ncols=1,
    figsize=(10, 6),
    labels=Labels(),
    padding=Padding(),
    fontsize: int = 12,
    **kwargs,
):
    """Create a matplotlib figure/axes pair with project-style defaults.

    Args:
        nrows (int): Number of subplot rows.
        ncols (int): Number of subplot columns.
        figsize (tuple): Figure size.
        labels (Labels): Global title/xlabel/ylabel container.
        padding (Padding): Layout padding parameters.
        fontsize (int): Base font size.
        **kwargs: Additional arguments forwarded to `plt.subplots`.

    Returns:
        tuple: `(fig, axes)` from matplotlib.
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
    """Save current matplotlib figure and optionally close it.

    Args:
        filename (str): Output filename stem.
        output_dir (Optional[str]): Destination directory.
        close (bool): Whether to close current figure.
        fmt (str): Figure format.

    Returns:
        None
    """

    if output_dir:
        check_and_create_dir(output_dir)
        savepath = f"{output_dir}/{filename}.{fmt}"
        plt.savefig(savepath, bbox_inches="tight", facecolor="w")
        LOGGER.info("Saved figure to %s.", savepath)

    if close:
        plt.close()


def clear_default_labels(ax) -> None:
    """Clear axis title and axis labels set by default helpers.

    Args:
        ax: Matplotlib axis.

    Returns:
        None
    """

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")


def add_escape_velocity(ax, v_max: float, y_max: float, log_xscale: bool = False) -> None:
    """Annotate host escape-velocity reference lines on an axis.

    Args:
        ax: Matplotlib axis.
        v_max (float): Maximum x-range value currently displayed.
        y_max (float): Maximum y-range value used for text placement.
        log_xscale (bool): Whether x-axis uses logarithmic scaling.

    Returns:
        None
    """

    colors = iter(mcolors.TABLEAU_COLORS.keys())
    # Plot vertical lines and labels (escape velocities)
    for label, v_esc in TypicalHostEscapeVelocity.latex_to_values().items():
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
    """Plot empirical PDF with median and credible-interval legend summary.

    Args:
        ax: Matplotlib axis.
        series (pd.Series): Input samples.
        color (str): Line color.
        name (Optional[str]): Label prefix.
        unit (Optional[str]): Optional unit suffix in legend text.
        ls (str): Line style.

    Returns:
        None
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
