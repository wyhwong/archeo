from typing import Optional

import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from archeo.visualization import base
import archeo.logger
from archeo.constants import EscapeVelocity, Columns as C
from archeo.schema import Labels, Padding


local_logger = archeo.logger.get_logger(__name__)


def mass_estimates(
    df: pd.DataFrame,
    label: str,
    filename: Optional[str] = None,
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

        filename (str):
            Output filename.

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

    padding = Padding(lpad=0.13, bpad=0.14)
    labels = Labels(
        title="Distribution of Estimated Masses",
        xlabel=r"Mass [$M_{\odot}$]",
        ylabel="PDF",
    )
    fig, ax = base.initialize_plot(figsize=(9, 4), labels=labels, padding=padding)
    colors = sns.color_palette("tab10")

    col_to_name = {
        C.REMNANT_BH_MASS: label + ": ",
        C.HEAVIER_BH_MASS: "Heavier Parent: ",
        C.LIGHTER_BH_MASS: "Ligher Parent: ",
    }
    for col, name in col_to_name.items():
        _plot_pdf(ax, next(colors), df[col], name, unit=r"[$M_{\odot}$]")

    _add_pisn_gap(ax, next(colors))
    ax.set(ylabel="", xlabel="")
    plt.legend()

    base.savefig_and_close(filename, output_dir, close)
    return (fig, ax)


def _add_pisn_gap(ax, color: str) -> None:
    """
    Add PISN gap to the plot.

    Args:
    -----
        ax (plt.Axes):
            Axes.

        color (str):
            Color.

    Returns:
    -----
        None
    """

    ax.axvline(65, color=color, linewidth=0.9, linestyle="--", label="PISN Gap")
    ax.axvline(130, color=color, linewidth=0.9, linestyle="--")


def corner_estimates(
    dfs: list[pd.DataFrame],
    labels: list[str],
    levels: Optional[list[float]] = None,
    nbins: int = 70,
    filename: Optional[str] = None,
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
        "full": [C.HEAVIER_BH_MASS, C.LIGHTER_BH_MASS, C.REMNANT_BH_MASS, C.REMNANT_RECOIL, C.SPIN_MAGNITUDE],
        "part": [C.HEAVIER_BH_MASS, C.LIGHTER_BH_MASS, C.REMNANT_RECOIL],
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
        colors = sns.color_palette("tab10")
        corner_labels = iter(labels)
        handles = []

        for df in dfs:

            if len(df) < nbins:
                local_logger.info("Dataframe does not have enough samples to plot.")
                continue

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


def second_generation_probability_curve(
    dfs: list[pd.DataFrame],
    labels: list[str],
    filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    close: bool = True,
):
    """
    Plot the second generation probability curve.

    Args:
    -----
        dfs (list[pd.DataFrame]):
            The posterior dataframes.

        label (list[str]):
            Label of each posterior.

        filename (str):
            Output filename.

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

    # Set up x-axis
    x_max = 0.0
    for df in dfs:
        if df[C.REMNANT_RECOIL].max() > x_max:
            x_max = df[C.REMNANT_RECOIL].max()
    x = np.linspace(0.0, x_max, 300)

    _padding = Padding(bpad=0.14)
    _labels = Labels(
        title="Second Generation Probability Curve",
        xlabel="Escape Velocity $v_{esc}$ [km s$^{-1}$]",
        ylabel="Second Generation Probability $p_{2g}$",
    )
    fig, ax = base.initialize_plot(figsize=(10, 8), labels=_labels, padding=_padding, fontsize=15)
    colors = iter(sns.color_palette("tab10"))

    for idx, df in enumerate(dfs):
        recovery_rate = df[C.RECOVERYRATE].iloc[0]
        # Calculate the CDF
        y = []
        for kick in x:
            df_samples = df.loc[
                (df[C.REMNANT_RECOIL] <= kick) & (df[C.HEAVIER_BH_MASS] <= 65) & (df[C.LIGHTER_BH_MASS] <= 65)
            ]
            if df_samples.empty:
                y.append(0.0)
            else:
                y.append(len(df_samples) / len(df) * recovery_rate)

        # Plot the CDF
        sns.lineplot(y=y, x=x, ax=ax, color=next(colors), label=labels[idx])

    _add_escape_velocity(ax, x_max)
    ax.set(ylabel="", xlabel="")
    plt.legend()

    base.savefig_and_close(filename, output_dir, close)
    return (fig, ax)


def _add_escape_velocity(ax, v_max: float) -> None:
    """
    Add escape velocity to the plot.

    Args:
    -----
        ax (plt.Axes):
            Axes.

        v_max (float):
            Maximum escape velocity.

    Returns:
    -----
        None
    """

    colors = sns.color_palette("tab10")
    # Plot vertical lines and labels (escape velocities)
    for label, v_esc in EscapeVelocity.to_vlines().items():
        # Skip if out of scope
        if v_esc > v_max:
            return

        color = next(colors)
        ax.axvline(x=v_esc, color=color, linestyle="--", linewidth=0.5)
        text_shift = 20.0 * v_max / 3000.0
        ax.text(v_esc + text_shift, 0.7, label, color=color, rotation=90, va="center", fontsize=12)


def effective_spin_estimates(
    dfs: list[pd.DataFrame],
    labels: list[str],
    filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    close: bool = True,
):
    """
    Plot the effective spin PDF.

    Args:
    -----
        dfs (list[pd.DataFrame]):
            The posterior dataframes.

        labels (list[str]):
            Label of each posterior.

        filename (str):
            Output filename.

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

    _padding = Padding(bpad=0.14)
    _labels = Labels(
        title="Effective Spin PDF",
        xlabel="Effective Spin $a_{eff}$",
        ylabel="PDF",
    )
    fig, ax = base.initialize_plot(figsize=(10, 8), labels=_labels, padding=_padding, fontsize=15)
    colors = sns.color_palette("tab10")

    for idx, df in enumerate(dfs):
        _get_effective_spin(df)
        _plot_pdf(ax, next(colors), df["a_eff"], labels[idx])

    plt.legend()
    ax.set(ylabel="", xlabel="")
    base.savefig_and_close(filename, output_dir, close)
    return (fig, ax)


def precession_spin_estimates(
    dfs: list[pd.DataFrame],
    labels: list[str],
    filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    close: bool = True,
):
    """
    Plot the precession spin PDF.

    Args:
    -----
        dfs (list[pd.DataFrame]):
            The posterior dataframes.

        labels (list[str]):
            Label of each posterior.

        filename (str):
            Output filename.

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

    _padding = Padding(bpad=0.14)
    _labels = Labels(
        title="Precession Spin PDF",
        xlabel="Precession Spin $a_p$",
        ylabel="PDF",
    )
    fig, ax = base.initialize_plot(figsize=(10, 8), labels=_labels, padding=_padding, fontsize=15)
    colors = archeo.schemas.visualization.Color.value_iter()

    for idx, df in enumerate(dfs):
        _get_precession_spin(df)
        _plot_pdf(ax, next(colors), df["ap"], labels[idx])

    plt.legend()
    ax.set(ylabel="", xlabel="")
    base.savefig_and_close(filename, output_dir, close)
    return (fig, ax)


def _plot_pdf(
    ax,
    color: str,
    series: pd.Series,
    name: str,
    unit: Optional[str] = None,
):
    """
    Plot the PDF of a parameter.

    Args:
    -----
        ax (plt.Axes):
            Axes.

        color (str):
            Color.

        series (pd.Series):
            Series (pdf).

        name (str):
            Name.

        unit (Optional[str]):
            Unit.

    Returns:
    -----
        None
    """

    density, bins = np.histogram(a=series, bins=70, density=True)
    low, mid, high = series.quantile(0.05), series.quantile(0.5), series.quantile(0.95)
    ax_label = "%s: $%.2f_{-%.2f}^{+%.2f}$" % (name, mid, mid - low, high - mid)
    if unit:
        ax_label += f" {unit}"
    ax.stairs(density, bins, label=ax_label, color=color)


def table_estimates(
    dfs: list[pd.DataFrame],
    labels: list[str],
    filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    close: bool = True,
):
    """
    Plot the posterior mass estimates.

    Args:
    -----
        dfs (list[pd.DataFrame]):
            The posterior dataframes.

        labels (list[str]):
            Label of each posterior.

        filename (str):
            Output filename.

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

    for df in dfs:
        _get_effective_spin(df)
        _get_precession_spin(df)

    col_to_names = {
        "m1": "$m_1$",
        "m2": "$m_2$",
        "q": "$q$",
        "mf_": "$m_f$",
        "chif": "$\\chi_f$",
        "vf": "$v_f$",
        "ap": "$a_{p}$",
        "a_eff": "$a_{eff}$",
    }
    data = {
        "": labels,
        "Recovery Rate": [df["recovery_rate"].iloc[0] for df in dfs],
    }

    for col, name in col_to_names.items():
        data[name] = []
        for df in dfs:
            low, mid, high = (
                df[col].quantile(0.05),
                df[col].quantile(0.5),
                df[col].quantile(0.95),
            )
            value = "$%.2f_{-%.2f}^{+%.2f}$" % (mid, mid - low, high - mid)
            data[name].append(value)

    df_table = pd.DataFrame(data)

    fig, ax = base.initialize_plot(figsize=(15, 4))
    ax.axis("off")
    ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc="center",
        loc="center",
    )

    base.savefig_and_close(filename, output_dir, close)
    return (fig, ax)


def _get_effective_spin(df: pd.DataFrame) -> None:
    """
    Get the effective spin.

    Args:
    -----
        df (pd.DataFrame):
            The posterior dataframe.

    Returns:
    -----
        None
    """

    a1z = df["a1"].apply(lambda x: x[-1])
    a2z = df["a1"].apply(lambda x: x[-1])
    df["a_eff"] = (df["m1"] * a1z + df["m2"] * a2z) / (df["m1"] + df["m2"])


def _get_precession_spin(df: pd.DataFrame) -> None:
    """
    Get the precession spin.

    Args:
    -----
        df (pd.DataFrame):
            The posterior dataframe.

    Returns:
    -----
        None
    """

    a1h = df["a1"].apply(lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2))
    a2h = df["a2"].apply(lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2))
    df["ap"] = np.maximum(a1h, (4 / df["q"] + 3) / (3 / df["q"] + 4) / df["q"] * a2h)
