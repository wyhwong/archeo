import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import corner

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


def savefig_and_close(plot_filename: str, output_dir=None, savefig=False, close=True) -> None:
    if savefig:
        if output_dir is None:
            raise ValueError(f"output_dir must not be empty if savefig is True.")
        check_and_create_dir(dirpath=output_dir)
        savepath = f"{output_dir}/{plot_filename}"
        plt.savefig(savepath, facecolor="w")
        LOGGER.debug(f"Saved plot at {savepath}.")
    if close:
        plt.close()
        LOGGER.debug("Closed plot.")


def plot_prior_params_distribution(prior_df: pd.DataFrame, output_dir=None, savefig=False, close=True) -> None:
    _, axes = initialize_plot(
        nrows=len(prior_df.columns), height=16, width=16, title="Distribution of remnant black-hole parameters"
    )
    for index, column in enumerate(prior_df.columns):
        sns.histplot(prior_df[column], ax=axes[index], element="step", fill=False, stat="density")
        LOGGER.debug(f"Plotted parameter ({column}) as histogram.")
    savefig_and_close(
        plot_filename="prior_params_distribution.png", output_dir=output_dir, savefig=savefig, close=close
    )


def plot_prior_kick_against_spin(prior_df: pd.DataFrame, output_dir=None, savefig=False, close=True) -> None:
    _, ax = initialize_plot(height=8, width=15, title="Natal kick against spin (Remnant black holes in prior)")

    x, y = prior_df["$\chi_f$"], prior_df["$v_f$"]
    sns.scatterplot(x=x, y=y, s=5, color=".15", ax=ax)
    LOGGER.debug("Plotted prior kick against spin as scatterplot.")
    savefig_and_close(
        plot_filename="prior_kick_against_spin_scatterplot.png", output_dir=output_dir, savefig=savefig, close=close
    )
    sns.histplot(x=x, y=y, bins=120, pthresh=0.05, ax=ax)
    LOGGER.info("Plotted prior kick against spin as 2D histogram.")
    sns.kdeplot(x=x, y=y, levels=[0.1, 0.3, 0.5], color="b", linewidths=1, ax=ax)
    LOGGER.info("Plotted prior kick against spin as kde plot.")
    savefig_and_close(plot_filename="prior_kick_against_spin.png", output_dir=output_dir, savefig=savefig, close=close)


def plot_prior_kick_distribution_on_spin(
    prior_df: pd.DataFrame,
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
        data = prior_df.loc[(spin_min_in_bin < prior_df["$\chi_f$"]) & (prior_df["$\chi_f$"] < spin_max_in_bin)][
            "$v_f$"
        ]
        if len(data.index) > 0:
            sns.histplot(
                data=data,
                ax=ax,
                element="step",
                fill=False,
                stat="density",
                label=f"$\chi_f$ $\in$ $[{spin_min_in_bin:.2f}, {spin_max_in_bin:.2f}]$",
            )
            LOGGER.debug(f"Plotted natal kick distribution, where {spin_min_in_bin} < spin < {spin_max_in_bin}.")
    plt.legend()
    savefig_and_close(
        plot_filename="prior_kick_distribution_on_spin.png", output_dir=output_dir, savefig=savefig, close=close
    )


def plot_parameter_estimation(
    prior_df: pd.DataFrame,
    target_parameter: str,
    target_parameter_label: str,
    posteriors: list,
    nbins: int = 200,
    plot_label: str = None,
    output_dir=None,
    savefig=False,
    close=True,
) -> None:
    _, ax = initialize_plot(
        height=8,
        width=15,
        title="Parameter Estimation",
        ylabel="Density",
        xlabel=f"Target parameter ({target_parameter_label})",
    )
    sns.histplot(
        prior_df[target_parameter],
        ax=ax,
        element="step",
        fill=False,
        stat="density",
        bins=nbins,
        label="Prior",
    )
    LOGGER.debug("Plotted prior, processing posteriors...")

    for posterior in posteriors:
        ax.stairs(values=posterior["values"], edges=posterior["edges"], label=posterior["label"])
        LOGGER.debug(f"Processed posterior {posterior['label']}")

    plt.legend()
    savefig_and_close(
        plot_filename=f"{plot_label}_{target_parameter}_estimation.png",
        output_dir=output_dir,
        savefig=savefig,
        close=close,
    )


def plot_posterior_corner(
    posterior_df: pd.DataFrame,
    posterior_label: str,
    var_names: list,
    labels: list,
    levels=[0.3, 0.9],
    output_dir=None,
    savefig=False,
    close=True,
) -> None:
    corner.corner(
        data=posterior_df,
        var_names=var_names,
        labels=labels,
        levels=levels,
    )
    savefig_and_close(
        plot_filename=f"{posterior_label}_corner_plot.png",
        output_dir=output_dir,
        savefig=savefig,
        close=close,
    )
