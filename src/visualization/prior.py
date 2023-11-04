import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import schemas.visualization
import visualization.base as base


def plot_dist(df: pd.DataFrame, output_dir=None, close=True) -> None:
    """
    Plot the distribution of the prior parameters.

    Parameters
    ----------

    df : pd.DataFrame
        Prior.
    output_dir : str, optional
        Output directory.
    close : bool, optional
        Whether to close the figure.

    Returns
    -------
    None
    """
    labels = schemas.visualization.Labels("Distribution of remnant black-hole parameters")
    _, axes = base.initialize_plot(len(df.columns), 1, (6, 8), labels)
    for index, column in enumerate(df.columns):
        sns.histplot(df[column], ax=axes[index], element="step", fill=False, stat="density")
    params_to_labels = [
        {"x": "Parent Mass Ratio $q$", "y": "PDF"},
        {"x": "Remnant Mass $m_f$ [$M_{\odot}$]", "y": "PDF"},
        {"x": "Recoil Kick $v_f$ [$kms^{-1}$]", "y": "PDF"},
        {"x": "Spin $\\chi_f$", "y": "PDF"},
    ]
    for index, param_to_label in enumerate(params_to_labels):
        axes[index].set(xlabel=param_to_label["x"], ylabel=param_to_label["y"])
    base.savefig_and_close("prior_dist.png", output_dir, close)


def plot_kick_against_spin(df: pd.DataFrame, output_dir=None, close=True) -> None:
    """
    Plot the distribution of the prior parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Prior.
    output_dir : str, optional
        Output directory.
    close : bool, optional
        Whether to close the figure.
    """
    padding = schemas.visualization.Padding(lpad=0.13)
    labels = schemas.visualization.Labels(
        "Remnant Kick against Remnant Spin",
        "Remnant Spin $\\chi_f$",
        "Remnant Kick $v_f [$kms^{-1}$]$",
    )
    _, ax = base.initialize_plot(figsize=(8, 6), labels=labels, padding=padding)

    sns.scatterplot(data=df, x="chif", y="vf", s=5, color=".15", ax=ax)
    ax.set(xlabel="", ylabel="")
    base.savefig_and_close("prior_kick_against_spin_scatter.png", output_dir, close)
    sns.histplot(data=df, x="chif", y="vf", bins=120, pthresh=0.05, ax=ax)
    sns.kdeplot(data=df, x="chif", y="vf", levels=[0.1, 0.3, 0.5], color="b", linewidths=1, ax=ax)
    ax.set(xlabel="", ylabel="")
    base.savefig_and_close("prior_kick_against_spin.png", output_dir, close)


def plot_kick_distribution_on_spin(df: pd.DataFrame, output_dir=None, close=True) -> None:
    """
    Plot the distribution of the prior parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Prior.
    output_dir : str, optional
        Output directory.
    close : bool, optional
        Whether to close the figure.

    Returns
    -------
    None
    """
    labels = schemas.visualization.Labels(
        "Remnant Kick Distribution on Different Spin Range", "Remnant Kick $v_f$ [$km/s$]", "PDF"
    )
    _, ax = base.initialize_plot(figsize=(9, 4), labels=labels)
    bounds = zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10))

    for low_bound, up_bound in bounds:
        data = df.loc[(low_bound < df["chif"]) & (df["chif"] < up_bound)]["vf"]
        # To avoid extreme density values
        if len(data.index) > 100:
            density, bins = np.histogram(a=data, density=True)
            ax.stairs(density, bins, label=f"$\chi_f$ $\in$ $[{low_bound:.2f}, {up_bound:.2f}]$")
    ax.set(xlabel="", ylabel="")
    plt.legend()
    base.savefig_and_close("prior_kick_on_spin.png", output_dir, close)
