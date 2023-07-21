import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import corner

from dataclasses import dataclass

from .common import check_and_create_dir
from .logger import get_logger


LOGGER = get_logger("utils | visualization")


@dataclass
class Padding:
    tpad: float = 2.5
    lpad: float = 0.1
    bpad: float = 0.12


@dataclass
class Labels:
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""


def initialize_plot(nrows=1, ncols=1, figsize=(10, 6), labels=Labels(), padding=Padding(), fontsize=12):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.tight_layout(pad=padding.tpad)
    fig.subplots_adjust(left=padding.lpad, bottom=padding.bpad)
    fig.suptitle(labels.title, fontsize=fontsize)
    fig.text(x=0.04, y=0.5, s=labels.ylabel, fontsize=fontsize, rotation="vertical", verticalalignment="center")
    fig.text(x=0.5, y=0.04, s=labels.xlabel, fontsize=fontsize, horizontalalignment="center")
    return fig, axes


def savefig_and_close(filename: str, output_dir=None, close=True) -> None:
    if output_dir:
        check_and_create_dir(output_dir)
        savepath = f"{output_dir}/{filename}"
        plt.savefig(savepath, facecolor="w")
        LOGGER.info(f"Saved plot at {savepath}.")
    if close:
        plt.close()


def plot_prior_params_distribution(prior_df: pd.DataFrame, output_dir=None, close=True) -> None:
    labels = Labels("Distribution of remnant black-hole parameters")
    _, axes = initialize_plot(nrows=len(prior_df.columns), figsize=(16, 16), labels=labels)
    for index, column in enumerate(prior_df.columns):
        sns.histplot(prior_df[column], ax=axes[index], element="step", fill=False, stat="density")
        LOGGER.debug(f"Plotted parameter ({column}) as histogram.")
    savefig_and_close("prior_params_distribution.png", output_dir, close)


def plot_prior_kick_against_spin(prior_df: pd.DataFrame, output_dir=None, close=True) -> None:
    labels = Labels("Natal kick against spin (Remnant black holes in prior)")
    _, ax = initialize_plot(figsize=(15, 8), labels=labels)

    x, y = prior_df["$\chi_f$"], prior_df["$v_f$"]
    sns.scatterplot(x=x, y=y, s=5, color=".15", ax=ax)
    LOGGER.debug("Plotted prior kick against spin as scatterplot.")
    savefig_and_close("prior_kick_against_spin_scatterplot.png", output_dir, close)
    sns.histplot(x=x, y=y, bins=120, pthresh=0.05, ax=ax)
    LOGGER.info("Plotted prior kick against spin as 2D histogram.")
    sns.kdeplot(x=x, y=y, levels=[0.1, 0.3, 0.5], color="b", linewidths=1, ax=ax)
    LOGGER.info("Plotted prior kick against spin as kde plot.")
    savefig_and_close("prior_kick_against_spin.png", output_dir, close)


def plot_prior_kick_distribution_on_spin(
    prior_df: pd.DataFrame,
    nbins: int,
    spin_max: float,
    spin_min: float,
    output_dir=None,
    close=True,
) -> None:
    labels = Labels("Natal kick distribution on different spin range (Remnant black holes in prior)")
    _, ax = initialize_plot(figsize=(15, 8), labels=labels)
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
    savefig_and_close("prior_kick_distribution_on_spin.png", output_dir, close)


def plot_parameter_estimation(
    prior_df: pd.DataFrame,
    target_parameter: str,
    target_parameter_label: str,
    likelihoods: list,
    nbins: int = 200,
    plot_label: str = None,
    output_dir=None,
    close=True,
) -> None:
    labels = Labels("Parameter Estimation", f"Target parameter ({target_parameter_label})", "Density")
    _, ax = initialize_plot(figsize=(15, 8), labels=labels)
    if prior_df is not None:
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
    else:
        LOGGER.debug("Prior not input, skipped, processing posteriors...")

    for likelihood in likelihoods:
        ax.stairs(values=likelihood["values"], edges=likelihood["edges"], label=likelihood["label"])
        LOGGER.debug(f"Processed posterior {likelihood['label']}")

    plt.ylabel(""), plt.xlabel("")
    plt.legend()
    savefig_and_close(f"{plot_label}_{target_parameter}_estimation.png", output_dir, close)


def plot_posterior_corner(
    posterior_df: pd.DataFrame,
    posterior_label: str,
    var_names: list,
    labels: list,
    levels=[0.68, 0.9],
    nbins=70,
    output_dir=None,
    close=True,
) -> None:
    corner.corner(
        data=posterior_df,
        var_names=var_names,
        labels=labels,
        levels=levels,
        plot_density=True,
        plot_samples=False,
        color="blue",
        fill_contours=False,
        smooth=True,
        bins=nbins,
        plot_datapoints=False,
        hist_kwargs=dict(density=True),
    )
    savefig_and_close(f"{posterior_label}_corner_plot.png", output_dir, close)
