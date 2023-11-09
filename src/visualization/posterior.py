import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import corner

import schemas.visualization
import visualization.base as base


def plot_mass_estimates(
    df: pd.DataFrame, label: str, output_dir=None, close=True
) -> None:
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
    labels = schemas.visualization.Labels(
        "Distribution of Estimated Masses", "Mass [$M_{\odot}$]", "PDF"
    )
    _, ax = base.initialize_plot(figsize=(9, 4), labels=labels, padding=padding)
    col_to_labels = {
        "mf_": f"{label}: ",
        "m1": "Heavier Parent: ",
        "m2": "Ligher Parent: ",
    }

    for col, label_prefix in col_to_labels.items():
        density, bins = np.histogram(a=df[col], bins=70, density=True)
        inv_low, med, inv_high = (
            df[col].quantile(0.05),
            df[col].quantile(0.5),
            df[col].quantile(0.95),
        )
        ax_label = "%s: $%d_{-%d}^{+%d}$ %s" % (
            label_prefix,
            med,
            med - inv_low,
            inv_high - med,
            "[$M_{\odot}$]",
        )
        ax.stairs(density, bins, label=ax_label)

    plt.ylabel(""), plt.xlabel("")
    plt.legend()
    base.savefig_and_close(f"{label}_mass_estimates.png", output_dir, close)


def plot_corner(
    df: pd.DataFrame,
    label: str,
    levels=[0.68, 0.9],
    nbins=70,
    output_dir=None,
    close=True,
) -> None:
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
        var_names=["m1", "m2", "mf_", "vf", "chif"],
        labels=["$m_1$", "$m_2$", "$m_f$", "$v_f$", "$\chi_f$"],
        levels=levels,
        plot_density=True,
        plot_samples=False,
        color="blue",
        fill_contours=False,
        smooth=True,
        plot_datapoints=False,
        hist_kwargs=dict(density=True),
    )
    plt.legend()
    base.savefig_and_close(f"{label}_corner.png", output_dir, close)


def plot_cumulative_kick_probability_curve(
    df: pd.DataFrame, label: str, output_dir=None, close=True
) -> None:
    """
    Plot the cumulative kick probability curve.

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
    padding = schemas.visualization.Padding(bpad=0.14)
    labels = schemas.visualization.Labels(
        "Cumulative Kick Probability Curve", "Recoil Velocity $v_f$ ($km/s$)", "CDF"
    )
    _, ax = base.initialize_plot(figsize=(9, 4), labels=labels, padding=padding)
    data = df.loc[(df["m1"] < 65) & (df["m2"] < 65)]
    norm_factor = len(data) / len(df)
    density, bins = np.histogram(data["vf"], 70, density=True)
    dbin = bins[1] - bins[0]
    density *= norm_factor
    cdf = np.cumsum(density) * dbin
    x = bins[:-1] + dbin / 2
    sns.lineplot(y=[0] + list(cdf), x=[0] + list(x), ax=ax, label=label)
    ax.set(ylabel="", xlabel="")
    plt.legend()
    base.savefig_and_close(
        f"{label}_cumulative_kick_probability_curve.png", output_dir, close
    )
