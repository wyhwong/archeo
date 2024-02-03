from typing import Optional

import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import core.visualization.base as base
import schemas.visualization


def mass_estimates(
    df: pd.DataFrame,
    label: str,
    output_dir: Optional[str] = None,
    close: bool = True,
):
    """
    Plot the posterior mass estimates.

    Args:
    -----
        df (pd.DataFrame):
            The posterior dataframe.

        label (str):
            Label of the posterior.

        output_dir (Optional[str]):
            Output directory.

        close (bool):
            Whether to close the figure.

    Returns:
    -----
        fig (plt.Figure):
            Figure.

        axes (plt.Axes):
            Axes.
    """

    padding = schemas.visualization.Padding(lpad=0.13, bpad=0.14)
    labels = schemas.visualization.Labels("Distribution of Estimated Masses", r"Mass [$M_{\odot}$]", "PDF")
    fig, ax = base.initialize_plot(figsize=(9, 4), labels=labels, padding=padding)
    colors = iter([color.value for color in schemas.visualization.Color])

    col_to_labels = {
        "mf_": f"{label}: ",
        "m1": "Heavier Parent: ",
        "m2": "Ligher Parent: ",
    }
    for col, label_prefix in col_to_labels.items():
        color = next(colors)
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
            r"[$M_{\odot}$]",
        )
        ax.stairs(density, bins, label=ax_label, color=color)

    color = next(colors)
    ax.axvline(65, color=color, linewidth=0.9, linestyle="--", label="PISN Gap")
    ax.axvline(130, color=color, linewidth=0.9, linestyle="--")

    ax.set(ylabel="", xlabel="")
    plt.legend()

    base.savefig_and_close(f"{label}_mass_estimates.png", output_dir, close)
    return (fig, ax)


def corner_estimates(
    dfs: list[pd.DataFrame],
    labels: list[str],
    levels: Optional[list[float]] = None,
    nbins: int = 70,
    filename: str = "corner.png",
    output_dir: Optional[str] = None,
    close: bool = True,
):
    """
    Plot the posterior corner plot.

    Args:
    -----
        dfs (list[pd.DataFrame]):
            The posterior dataframes.

        labels (list[str]):
            Label of each posterior.

        levels (list[float]):
            Contour levels.

        nbins (int):
            Number of bins.

        filename (str):
            Output filename.

        output_dir (Optional[str]):
            Output directory.

        close (bool):
            Whether to close the figure.

    Returns:
    -----
        None
    """

    if not levels:
        levels = [0.68, 0.9]

    corner_type_to_var_names = {
        "full": ["m1", "m2", "mf_", "vf", "chif"],
        "part": ["m1", "m2", "vf"],
    }
    corner_type_to_labels = {
        "full": [
            r"$m_1$ [$M_{\odot}$]",
            r"$m_2$ [$M_{\odot}$]",
            r"$m_f$ [$M_{\odot}$]",
            r"$v_f$ [km s$^{-1}$]",
            "$\\chi_f$",
        ],
        "part": [r"$m_1$ [$M_{\odot}$]", r"$m_2$ [$M_{\odot}$]", r"$v_f$ [km s$^{-1}$]"],
    }

    for corner_type, var_names in corner_type_to_var_names.items():
        fig, _ = base.initialize_plot(ncols=len(var_names), nrows=len(var_names), figsize=(9, 9))
        colors = iter([color.value for color in schemas.visualization.Color])
        corner_labels = iter(labels)
        handles = []

        for df in dfs:
            color = next(colors)
            corner.corner(
                data=df[var_names],
                bins=nbins,
                var_names=var_names,
                labels=corner_type_to_labels[corner_type],
                levels=levels,
                plot_density=True,
                plot_samples=False,
                color=color,
                fill_contours=False,
                smooth=True,
                plot_datapoints=False,
                hist_kwargs=dict(density=True),
                fig=fig,
            )
            handles.append(mlines.Line2D([], [], color=color, label=next(corner_labels)))

        fig.legend(
            handles=handles,
            fontsize=15,
            frameon=True,
            bbox_to_anchor=(1.0, 0.95),
            loc="upper right",
        )
        base.savefig_and_close(f"{corner_type}_{filename}", output_dir, close)


def cumulative_kick_probability_curve(
    dfs: list[pd.DataFrame],
    labels: list[str],
    xlim: Optional[tuple[float, float]] = None,
    filename: str = "cumulative_kick_probability_curve.png",
    output_dir: Optional[str] = None,
    close: bool = True,
):
    """
    Plot the cumulative kick probability curve.

    Args:
    -----
        dfs (list[pd.DataFrame]):
            The posterior dataframes.

        label (list[str]):
            Label of each posterior.

        xlim (Optional[tuple[float, float]]):
            X-axis limits.

        output_dir (Optional[str]):
            Output directory.

        close (bool):
            Whether to close the figure.

    Returns:
    -----
        fig (plt.Figure):
            Figure.

        axes (plt.Axes):
            Axes.
    """

    if not xlim:
        xlim = (0.0, 3000.0)

    padding = schemas.visualization.Padding(bpad=0.14)
    plot_labels = schemas.visualization.Labels(
        title="Cumulative Kick Probability Curve",
        xlabel="Recoil Velocity $v_f$ [km s$^{-1}$]",
        ylabel="CDF",
    )
    fig, ax = base.initialize_plot(figsize=(10, 8), labels=plot_labels, padding=padding, fontsize=15)
    v_values = [
        schemas.binary.EscapeVelocity.GLOBULAR_CLUSTER.value,
        schemas.binary.EscapeVelocity.MILKY_WAY.value,
        schemas.binary.EscapeVelocity.NUCLEAR_STAR_CLUSTER.value,
        schemas.binary.EscapeVelocity.ELLIPTICAL_GALAXY.value,
    ]
    v_labels = [
        "$v_{esc}$ Globular Cluster",
        "$v_{esc}$ Milky Way",
        "$v_{esc}$ Nuclear Star Cluster",
        "$v_{esc}$ (Elliptical Galaxy)",
    ]
    v_colors = ["brown", "black", "red", "purple"]

    colors = iter([color.value for color in schemas.visualization.Color])
    x = np.linspace(xlim[0], xlim[1], 1000)
    h_indices = [np.abs(x - v_value).argmin() for v_value in v_values]

    # Plot vertical lines and labels (escape velocities)
    for idx, v_value in enumerate(v_values):
        # Skip if out of scope
        if v_value > xlim[1]:
            break

        ax.axvline(x=v_value, color=v_colors[idx], linestyle="--", linewidth=0.5)
        shift = 20.0 * xlim[1] / 3000.0
        ax.text(v_value + shift, 0.7, v_labels[idx], color=v_colors[idx], rotation=90, va="center", fontsize=12)

    potential_yticks = []
    for idx_1, df in enumerate(dfs):
        # Calculate the CDF
        y = []
        for kick in x:
            df_samples = df.loc[(df["vf"] <= kick) & (df["m1"] <= 65) & (df["m2"] <= 65)]
            if df_samples.empty:
                y.append(0.0)
            else:
                y.append(len(df_samples) / len(df))

        # Plot horizontal lines (intersection with escape velocities)
        for idx_2, v_value in enumerate(v_values):
            # Skip if out of scope
            if v_value > xlim[1]:
                break

            # Horizontal lines at intersection
            intersection = y[h_indices[idx_2]]
            ax.axhline(y=intersection, color=v_colors[idx_2], linestyle="--", linewidth=0.5)
            potential_yticks.append(intersection)

        color = next(colors)
        # Plot the CDF
        sns.lineplot(y=y, x=x, ax=ax, color=color, label=labels[idx_1])

    # Set y-ticks (this step is to avoid overlapping y-ticks)
    potential_yticks = sorted(potential_yticks)
    yticks = [potential_yticks[0]]
    for potential_ytick in potential_yticks:
        if (potential_ytick - yticks[-1]) > 0.05:
            yticks.append(potential_ytick)
    ax.set_yticks(yticks)

    ax.set(ylabel="", xlabel="", xlim=xlim)
    plt.legend()

    base.savefig_and_close(filename, output_dir, close)
    return (fig, ax)
