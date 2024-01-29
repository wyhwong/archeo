import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

import utils.common
import schemas.visualization


def initialize_plot(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[int, int] = (10, 6),
    labels: schemas.visualization.Labels = schemas.visualization.Labels(),
    padding: schemas.visualization.Padding = schemas.visualization.Padding(),
    fontsize: int = 12,
) -> tuple[plt.Figure, np.ndarray[plt.Axes] | plt.Axes]:
    """
    Initialize the plot.

    Parameters
    ----------
    nrows : int, optional
        Number of rows of the plot.
    ncols : int, optional
        Number of columns of the plot.
    figsize : tuple, optional
        Size of the figure.
    labels : Labels, optional
        Labels for the plot.
    padding : Padding, optional
        Padding for the plot.
    fontsize : int, optional
        Fontsize of the labels.

    Returns
    -------
    fig : Figure
        Figure of the plot.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
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
    fig.text(
        x=0.5, y=0.04, s=labels.xlabel, fontsize=fontsize, horizontalalignment="center"
    )
    return (fig, axes)


def savefig_and_close(
    filename: Optional[str] = None, output_dir: Optional[str] = None, close: bool = True
) -> None:
    """
    Save the figure and close it.

    Parameters
    ----------
    filename : str, optional
        Filename to save the figure.
    output_dir : str, optional
        Output directory to save the figure.
    close : bool, optional
        Whether to close the figure.

    Returns
    -------
    None
    """
    if output_dir:
        utils.common.check_and_create_dir(output_dir)
        savepath = f"{output_dir}/{filename}"
        plt.savefig(savepath, facecolor="w")
    if close:
        plt.close()
