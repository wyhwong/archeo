import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import corner

import schemas.visualization
import visualization.base as base


def plot_mass_estimates(df: pd.DataFrame, label: str, output_dir=None, close=True) -> None:
    """
    Plot the distribution of the estimated masses.

    Parameters
    ----------
    df : pd.DataFrame
        Posterior.
    label : str
        Posterior label.
    output_dir : str, optional
        Output directory.
    close : bool, optional
        Whether to close the figure.

    Returns
    -------
    None
    """
    padding = schemas.visualization.Padding(lpad=0.13, bpad=0.14)
    labels = schemas.visualization.Labels("Distribution of Estimated Masses", "Mass $(M_{\odot})$", "Density")
    _, ax = base.initialize_plot(figsize=(9, 4), labels=labels, padding=padding)
    col_to_labels = {"mf": f"{label}: ", "m_p1": "Heavier Parent: ", "m_p2": "Ligher Parent: "}

    for col, label_prefix in col_to_labels.items():
        density, bins = np.histogram(a=df[col], density=True)
        inv_low, med, inv_high = df[col].quantile(0.05), df[col].quantile(0.5), df[col].quantile(0.95)
        ax_label = "%s: $%d_{-%d}^{+%d}$ %s" % (label_prefix, med, inv_low, inv_high, "($M_{\odot}$)")
        ax.stairs(density, bins, label=ax_label)

    plt.ylabel(""), plt.xlabel("")
    plt.legend()
    base.savefig_and_close(f"{label}_mass_estimates.png", output_dir, close)


def plot_corner(df: pd.DataFrame, label: str, levels=[0.68, 0.9], nbins=70, output_dir=None, close=True) -> None:
    """
    Plot the posterior corner plot.

    Parameters
    ----------
    df : pd.DataFrame
        Posterior.
    label : str
        Posterior label.
    levels : list, optional
        The contour levels, by default [0.68, 0.9].
    nbins : int, optional
        The number of bins, by default 70.
    output_dir : str, optional
        Output directory.
    close : bool, optional
        Whether to close the figure.

    Returns
    -------
    None
    """
    corner.corner(
        df,
        nbins,
        var_names=["m_p1", "m_p2", "mf", "vf", "chif"],
        labels=["$m_p1$", "$m_p2$", "$m_f$", "$v_f$", "$\chi_f$"],
        levels=levels,
        plot_density=True,
        plot_samples=False,
        color="blue",
        fill_contours=False,
        smooth=True,
        plot_datapoints=False,
        hist_kwargs=dict(density=True),
    )
    base.savefig_and_close(f"{label}_corner.png", output_dir, close)


def plot_cumulative_kick_probability_curve(
    df: pd.DataFrame, label: str, include_pisn=False, output_dir=None, close=True
) -> None:
    """
    Plot the cumulative kick probability curve.

    Parameters
    ----------
    df : pd.DataFrame
        Posterior.
    label : str
        Posterior label.
    include_pisn : bool, optional
        Whether to include samples in PISN gap, by default False.
    output_dir : str, optional
        Output directory.
    close : bool, optional
        Whether to close the figure.

    Returns
    -------
    None
    """
    padding = schemas.visualization.Padding(bpad=0.14)
    labels = schemas.visualization.Labels(
        "Cumulative Kick Probability Curve", "Recoil Velocity $v_f$ ($km/s$)", "Cumulative Probability"
    )
    _, ax = base.initialize_plot(figsize=(9, 4), labels=labels, padding=padding)
    if include_pisn:
        data = df
    else:
        data = df.loc[(df["m_p1"] < 65.0) & (df["m_p2"] < 65.0)]

    sns.kdeplot(data=data["vf"], cut=0, ax=ax, cumulative=True, label=label)
    ax.set(ylabel="", xlabel="")
    plt.legend()
    base.savefig_and_close(f"{label}_cumulative_kick_probability_curve.png", output_dir, close)
