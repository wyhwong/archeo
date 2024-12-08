from typing import Optional

import corner
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import archeo.logger
from archeo.constants import Columns as C
from archeo.constants import EscapeVelocity
from archeo.schema import Labels, Padding
from archeo.visualization import base


local_logger = archeo.logger.get_logger(__name__)


def mass_estimates(
    df: pd.DataFrame,
    label: str,
    filename="mass_estimates",
    output_dir: Optional[str] = None,
    close: bool = True,
    fmt: str = "png",
):
    """Plot the posterior mass estimates.

    Args:
        df (pd.DataFrame): The posterior dataframe.
        label (str): Label of the posterior.
        filename (str): Output filename.
        output_dir (Optional[str]): Output directory.
        close (bool): Whether to close the figure.
        fmt (str): The format of the visualizations. Defaults to "png".

    Returns:
        fig (plt.Figure): Figure.
        axes (plt.Axes): Axes.
    """

    padding = Padding(lpad=0.13, bpad=0.14)
    labels = Labels(
        title="Distribution of Estimated Masses",
        xlabel=r"Mass [$M_{\odot}$]",
        ylabel="PDF",
    )
    fig, ax = base.initialize_plot(figsize=(9, 4), labels=labels, padding=padding)
    colors = iter(mcolors.TABLEAU_COLORS.keys())

    col_to_name = {
        C.BH_MASS: label + ": ",
        C.HEAVIER_BH_MASS: "Heavier Parent: ",
        C.LIGHTER_BH_MASS: "Ligher Parent: ",
    }
    for col, name in col_to_name.items():
        _plot_pdf(ax, next(colors), df[col], name, unit=r"[$M_{\odot}$]")
    _add_pisn_gap(ax, next(colors))

    ax.set(ylabel="", xlabel="")
    plt.legend()
    base.savefig_and_close(filename, output_dir, close, fmt)
    return (fig, ax)


def _add_pisn_gap(ax, color: str) -> None:
    """Add PISN gap to the plot.

    Args:
        ax (plt.Axes): Axes.
        color (str): Color.
    """

    ax.axvline(65.0, color=color, linewidth=0.9, linestyle="--", label="PISN Gap")
    ax.axvline(130.0, color=color, linewidth=0.9, linestyle="--")


def corner_estimates(  # pylint: disable=dangerous-default-value
    dfs: dict[str, pd.DataFrame],
    levels: list[float] = [0.68, 0.9],
    nbins: int = 70,
    filename="corner_estimates",
    output_dir: Optional[str] = None,
    close: bool = True,
    fmt: str = "png",
):
    """Plot the posterior corner plot.

    Args:
        dfs (dict[str, pd.DataFrame]): Key: name of the posterior, value: posterior dataframe.
        levels (list[float]): Contour levels.
        nbins (int): Number of bins.
        filename (str): Output filename.
        output_dir (Optional[str]): Output directory.
        close (bool): Whether to close the figure.
        fmt (str): The format of the visualizations. Defaults to "png".

    Returns:
        fig (plt.Figure): Figure.
        axes (plt.Axes): Axes.
    """

    corner_type_to_var_names = {
        "part": [C.HEAVIER_BH_MASS, C.LIGHTER_BH_MASS, C.BH_KICK],
        "full": [C.HEAVIER_BH_MASS, C.LIGHTER_BH_MASS, C.BH_MASS, C.BH_KICK, C.BH_SPIN],
    }
    corner_type_to_labels = {
        "part": [r"$m_1$ [$M_{\odot}$]", r"$m_2$ [$M_{\odot}$]", r"$v_f$ [km s$^{-1}$]"],
        "full": [
            r"$m_1$ [$M_{\odot}$]",
            r"$m_2$ [$M_{\odot}$]",
            r"$m_f$ [$M_{\odot}$]",
            r"$v_f$ [km s$^{-1}$]",
            "$\\chi_f$",
        ],
    }

    for corner_type, var_names in corner_type_to_var_names.items():
        fig, axes = base.initialize_plot(ncols=len(var_names), nrows=len(var_names), figsize=(9, 9))
        colors = iter(mcolors.TABLEAU_COLORS.keys())
        handles = []

        for label, df in dfs.items():
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
            handles.append(mlines.Line2D([], [], color=color, label=label))

        fig.legend(
            handles=handles,
            fontsize=15,
            frameon=True,
            bbox_to_anchor=(1.0, 0.95),
            loc="upper right",
        )
        base.savefig_and_close(f"{corner_type}_{filename}", output_dir, close, fmt)

    return (fig, axes)


def second_generation_probability_curve(
    dfs: dict[str, pd.DataFrame],
    filename="second_generation_probability_curve",
    output_dir: Optional[str] = None,
    close: bool = True,
    fmt: str = "png",
):
    """Plot the second generation probability curve.

    Args:
        dfs (list[pd.DataFrame]): Key: name of the posterior, value: posterior dataframe.
        label (list[str]): Label of each posterior.
        filename (str): Output filename.
        output_dir (Optional[str]): Output directory.
        close (bool): Whether to close the figure.
        fmt (str): The format of the visualizations. Defaults to "png".

    Returns:
        fig (plt.Figure): Figure.
        axes (plt.Axes): Axes.
    """

    # Set up x-axis
    x_max = max(df[C.BH_KICK].max() for df in dfs.values())
    x = np.linspace(0.0, x_max, 300)

    padding = Padding(bpad=0.14)
    labels = Labels(
        title="Second Generation Probability Curve",
        xlabel="Escape Velocity $v_{esc}$ [km s$^{-1}$]",
        ylabel="Second Generation Probability $p_{2g}$",
    )
    fig, ax = base.initialize_plot(figsize=(10, 8), labels=labels, padding=padding, fontsize=15)
    colors = iter(mcolors.TABLEAU_COLORS.keys())

    for label, df in dfs.items():
        recovery_rate = df[C.RECOVERY_RATE].iloc[0]
        # Calculate the CDF
        y = []
        for kick in x:
            df_samples = df.loc[(df[C.BH_KICK] <= kick) & (df[C.HEAVIER_BH_MASS] <= 65) & (df[C.LIGHTER_BH_MASS] <= 65)]
            if df_samples.empty:
                y.append(0.0)
            else:
                y.append(len(df_samples) / len(df) * recovery_rate)

        # Plot the CDF
        sns.lineplot(y=y, x=x, ax=ax, color=next(colors), label=label)

    _add_escape_velocity(ax, x_max, max(y))
    ax.set(ylabel="", xlabel="")
    plt.legend()

    base.savefig_and_close(filename, output_dir, close, fmt)
    return (fig, ax)


def _add_escape_velocity(ax, v_max: float, y_max: float) -> None:
    """Add escape velocity to the plot.

    Args:
        ax (plt.Axes): Axes.
        v_max (float): Maximum escape velocity.
        y_max (float): Maximum y-axis value.
    """

    colors = iter(mcolors.TABLEAU_COLORS.keys())
    # Plot vertical lines and labels (escape velocities)
    for label, v_esc in EscapeVelocity.to_vlines().items():
        # Skip if out of scope
        if v_esc > v_max:
            return

        color = next(colors)
        ax.axvline(x=v_esc, color=color, linestyle="--", linewidth=0.5)
        text_shift = 20.0 * v_max / 3000.0
        ax.text(v_esc + text_shift, 0.7 * y_max, label, color=color, rotation=90, va="center", fontsize=12)


def effective_spin_estimates(
    dfs: dict[str, pd.DataFrame],
    filename="effective_spin_estimates",
    output_dir: Optional[str] = None,
    close: bool = True,
    fmt: str = "png",
):
    """Plot the effective spin PDF.

    Args:
        dfs (dict[str, pd.DataFrame]): Key: name of the posterior, value: posterior dataframe.
        filename (str): Output filename.
        output_dir (Optional[str]): Output directory.
        close (bool): Whether to close the figure.
        fmt (str): The format of the visualizations. Defaults to "png".

    Returns:
        fig (plt.Figure): Figure.
        axes (plt.Axes): Axes.
    """

    padding = Padding(bpad=0.14)
    labels = Labels(
        title="Effective Spin PDF",
        xlabel="Effective Spin $a_{eff}$",
        ylabel="PDF",
    )
    fig, ax = base.initialize_plot(figsize=(10, 8), labels=labels, padding=padding, fontsize=15)
    colors = iter(mcolors.TABLEAU_COLORS.keys())

    for label, df in dfs.items():
        _plot_pdf(ax, next(colors), df[C.BH_EFF_SPIN], label)

    plt.legend()
    ax.set(ylabel="", xlabel="")
    base.savefig_and_close(filename, output_dir, close, fmt)
    return (fig, ax)


def precession_spin_estimates(
    dfs: dict[str, pd.DataFrame],
    filename="precession_spin_estimates",
    output_dir: Optional[str] = None,
    close: bool = True,
    fmt: str = "png",
):
    """Plot the precession spin PDF.

    Args:
        dfs (dict[str, pd.DataFrame]): Key: name of the posterior, value: posterior dataframe.
        filename (str): Output filename.
        output_dir (Optional[str]): Output directory.
        close (bool): Whether to close the figure.
        fmt (str): The format of the visualizations. Defaults to "png".

    Returns:
        fig (plt.Figure): Figure.
        axes (plt.Axes): Axes.
    """

    padding = Padding(bpad=0.14)
    labels = Labels(
        title="Precession Spin PDF",
        xlabel="Precession Spin $a_p$",
        ylabel="PDF",
    )
    fig, ax = base.initialize_plot(figsize=(10, 8), labels=labels, padding=padding, fontsize=15)
    colors = iter(mcolors.TABLEAU_COLORS.keys())

    for label, df in dfs.items():
        _plot_pdf(ax, next(colors), df[C.BH_PREC_SPIN], label)

    plt.legend()
    ax.set(ylabel="", xlabel="")
    base.savefig_and_close(filename, output_dir, close, fmt)
    return (fig, ax)


def _plot_pdf(
    ax,
    color: str,
    series: pd.Series,
    name: str,
    unit: Optional[str] = None,
):
    """Plot the PDF of a parameter.

    Args:
        ax (plt.Axes): Axes.
        color (str): Color.
        series (pd.Series): Series (pdf).
        name (str): Name.
        unit (Optional[str]): Unit.
    """

    density, bins = np.histogram(a=series, bins=70, density=True)
    low, mid, high = series.quantile(0.05), series.quantile(0.5), series.quantile(0.95)
    label = "%s: $%.2f_{-%.2f}^{+%.2f}$" % (name, mid, mid - low, high - mid)
    if unit:
        label += f" {unit}"
    ax.stairs(density, bins, label=label, color=color)


def table_estimates(
    dfs: dict[str, pd.DataFrame],
    filename="table_estimates",
    output_dir: Optional[str] = None,
    close: bool = True,
    fmt: str = "png",
):
    """Plot the posterior mass estimates.

    Args:
        dfs (dict[str, pd.DataFrame]): Key: name of the posterior, value: posterior dataframe.
        filename (str): Output filename.
        output_dir (Optional[str]): Output directory.
        close (bool): Whether to close the figure.
        fmt (str): The format of the visualizations. Defaults

    Returns:
        fig (plt.Figure): Figure.
        axes (plt.Axes): Axes.
    """

    col_to_names = {
        C.HEAVIER_BH_MASS: "$m_1$",
        C.LIGHTER_BH_MASS: "$m_2$",
        C.MASS_RATIO: "$q$",
        C.BH_MASS: "$m_f$",
        C.BH_SPIN: "$\\chi_f$",
        C.BH_KICK: "$v_f$",
        C.BH_PREC_SPIN: "$a_{p}$",
        C.BH_EFF_SPIN: "$a_{eff}$",
    }
    data = {
        "": dfs.keys(),
        "Recovery Rate": [df[C.RECOVERY_RATE].iloc[0] for df in dfs.values()],
    }

    for col, name in col_to_names.items():
        data[name] = []
        for df in dfs.values():
            low, mid, high = (
                df[col].quantile(0.05),
                df[col].quantile(0.5),
                df[col].quantile(0.95),
            )
            value = "$%.2f_{-%.2f}^{+%.2f}$" % (mid, mid - low, high - mid)
            data[name].append(value)

    df_table = pd.DataFrame(data)

    fig, ax = base.initialize_plot(figsize=(15, 4), dpi=1200)
    ax.axis("off")
    ax.table(cellText=df_table.values, colLabels=df_table.columns, cellLoc="center", loc="center")
    base.savefig_and_close(filename, output_dir, close, fmt)
    return (fig, ax)
