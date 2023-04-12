import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import check_and_create_dir
from .logger import get_logger

LOGGER = get_logger(logger_name="Utils | Visualization")


def initialize_plot(
    nrows=1, ncols=1, height=6, width=10, title="", ylabel="", xlabel="", tpad=2.5, lpad=0.1, bpad=0.12, fontsize=12
):
    LOGGER.debug(
        f"Initializing plot: {nrows=}, {ncols=}, {height=}, {width=}, {title=}, {ylabel=}, {xlabel=}, {tpad=}, {lpad=}, {bpad=}, {fontsize=}..."
    )
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    fig.tight_layout(pad=tpad)
    fig.subplots_adjust(left=lpad, bottom=bpad)
    fig.suptitle(title, fontsize=fontsize)
    fig.text(x=0.04, y=0.5, s=ylabel, fontsize=fontsize, rotation="vertical", verticalalignment="center")
    fig.text(x=0.5, y=0.04, s=xlabel, fontsize=fontsize, horizontalalignment="center")
    LOGGER.debug("Initialized blank plot.")
    return fig, axes


def plot_prior_params_distribution(dataframe: pd.DataFrame, output_dir=None, savefig=False, close=True):
    _, axes = initialize_plot(
        nrows=len(dataframe.columns), height=16, width=16, title="Distribution of remnant black-hole parameters"
    )
    for index, column in enumerate(dataframe.columns):
        sns.histplot(dataframe[column], ax=axes[index], element="step", fill=False, stat="density")
        LOGGER.debug(f"Plotted parameter ({column}) as histogram.")
    if savefig:
        if output_dir is None:
            raise ValueError(f"output_dir ({output_dir}) must not be empty if savefig is True.")
        check_and_create_dir(dirpath=output_dir)
        savepath = f"{output_dir}/prior_params_distribution.png"
        plt.savefig(savepath, facecolor="w")
        LOGGER.debug(f"Saved prior parameter distribution at {savepath}.")
    if close:
        plt.close()
        LOGGER.debug("Closed plot.")


def plot_prior_kick_against_spin(dataframe: pd.DataFrame, output_dir=None, savefig=False, close=True):
    _, ax = initialize_plot(height=8, width=15, title="Natal kick against spin (Remnant black holes in prior)")

    x, y = dataframe["$\chi_f$"], dataframe["$v_f$"]
    sns.scatterplot(x=x, y=y, s=5, color=".15", ax=ax)
    LOGGER.debug("Plotted prior kick against spin as scatterplot.")
    if savefig:
        if output_dir is None:
            raise ValueError(f"output_dir ({output_dir}) must not be empty if savefig is True.")
        check_and_create_dir(dirpath=output_dir)
        savepath = f"{output_dir}/kick_against_spin_scatterplot.png"
        plt.savefig(savepath, facecolor="w")
        LOGGER.debug(f"Saved scatterplot (prior kick against spin) at {savepath}.")

    sns.histplot(x=x, y=y, bins=120, pthresh=0.05, ax=ax)
    LOGGER.info("Plotted prior kick against spin as 2D histogram.")
    sns.kdeplot(x=x, y=y, levels=[0.1, 0.3, 0.5], color="b", linewidths=1, ax=ax)
    LOGGER.info("Plotted prior kick against spin as kde plot.")
    if savefig:
        savepath = f"{output_dir}/prior_kick_against_spin.png"
        plt.savefig(savepath, facecolor="w")
        LOGGER.debug(f"Saved plot (prior kick against spin) at {savepath}.")
    if close:
        plt.close()
        LOGGER.debug("Closed plot.")


def plot_prior_kick_distribution_on_spin(
    dataframe: pd.DataFrame,
    nbins: int,
    spin_max: float,
    spin_min: float,
    output_dir=None,
    savefig=False,
    close=True,
) -> None:
    _, ax = initialize_plot(
        height=8, width=15, title="Natal kick distribution on different spin range (Remnant black holes in prior)"
    )
    spin_boundaries = np.linspace(spin_min, spin_max, nbins + 1)
    for index in range(len(spin_boundaries) - 1):
        spin_min_in_bin, spin_max_in_bin = spin_boundaries[index], spin_boundaries[index + 1]
        sns.histplot(
            data=dataframe.loc[(spin_min_in_bin < dataframe["$\chi_f$"]) & (dataframe["$\chi_f$"] < spin_max_in_bin)][
                "$v_f$"
            ],
            ax=ax,
            element="step",
            fill=False,
            stat="density",
            label=f"$\chi_f$ $\in$ $[{spin_min_in_bin:.2f}, {spin_max_in_bin:.2f}]$",
        )
        LOGGER.debug(f"Plotted natal kick distribution, where {spin_min_in_bin} < spin < {spin_max_in_bin}.")
    plt.legend()
    if savefig:
        if output_dir is None:
            raise ValueError(f"output_dir ({output_dir}) must not be empty if savefig is True.")
        savepath = f"{output_dir}/prior_kick_distribution_on_spin.png"
        plt.savefig(savepath, facecolor="w")
        LOGGER.debug(f"Saved plot at {savepath}.")
    if close:
        plt.close()
        LOGGER.debug("Closed plot.")
