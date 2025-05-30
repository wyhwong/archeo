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
from archeo.constants import Suffixes as S
from archeo.schema import Labels, Padding
from archeo.visualization import base


local_logger = archeo.logger.get_logger(__name__)


def filter_unmapped_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out the samples that are not mapped to the posterior.

    Args:
        df (pd.DataFrame): The prior dataframe.

    Returns:
        df (pd.DataFrame): The filtered prior dataframe.
    """

    return df.dropna(subset=[C.KICK])


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

    df = filter_unmapped_samples(df)

    padding = Padding(lpad=0.13, bpad=0.14)
    labels = Labels(
        title="Distribution of Estimated Masses",
        xlabel=r"Mass [$M_{\odot}$]",
        ylabel="PDF",
    )
    fig, ax = base.initialize_plot(figsize=(9, 4), labels=labels, padding=padding)
    colors = iter(mcolors.TABLEAU_COLORS.keys())

    col_to_name = {
        S.FINAL(C.MASS): label + ": ",
        S.PRIMARY(C.MASS): "Heavier Parent: ",
        S.SECONDARY(C.MASS): "Ligher Parent: ",
    }
    for col, name in col_to_name.items():
        _plot_pdf(ax, next(colors), df[col], name, unit=r"[$M_{\odot}$]")
    _add_pisn_gap(ax, next(colors))

    plt.legend()
    base.clear_default_labels(ax)
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
    with_precession: bool = False,
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
        with_precession (bool): Whether to include precession spin in the corner plot.
        filename (str): Output filename.
        output_dir (Optional[str]): Output directory.
        close (bool): Whether to close the figure.
        fmt (str): The format of the visualizations. Defaults to "png".

    Returns:
        fig (plt.Figure): Figure.
        axes (plt.Axes): Axes.
    """

    corner_type_to_var_names = {
        "part": [S.PRIMARY(C.MASS), S.SECONDARY(C.MASS), C.KICK],
        "full": [S.PRIMARY(C.MASS), S.SECONDARY(C.MASS), S.FINAL(C.MASS), C.KICK, S.FINAL(C.SPIN_MAG), S.EFF(C.SPIN)],
    }
    corner_type_to_labels = {
        "part": [r"$m_1$ [$M_{\odot}$]", r"$m_2$ [$M_{\odot}$]", r"$v_f$ [km s$^{-1}$]"],
        "full": [
            r"$m_1$ [$M_{\odot}$]",
            r"$m_2$ [$M_{\odot}$]",
            r"$m_f$ [$M_{\odot}$]",
            r"$v_f$ [km s$^{-1}$]",
            "$\\chi_f$",
            "$\\chi_{eff}$",
        ],
    }

    if with_precession:
        corner_type_to_var_names["full"].append(S.PREC(C.SPIN))
        corner_type_to_labels["full"].append("$\\chi_{p}$")

    for corner_type, var_names in corner_type_to_var_names.items():
        fig, axes = base.initialize_plot(ncols=len(var_names), nrows=len(var_names), figsize=(9, 9))
        colors = iter(mcolors.TABLEAU_COLORS.keys())
        handles = []

        for label, df in dfs.items():
            df = filter_unmapped_samples(df)

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
    x_max = max(df[C.KICK].max() for df in dfs.values())
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
        df = filter_unmapped_samples(df)

        # Calculate the CDF
        y = []
        for kick in x:
            df_samples = df.loc[(df[C.KICK] <= kick) & (df[S.PRIMARY(C.MASS)] <= 65) & (df[S.SECONDARY(C.MASS)] <= 65)]
            if df_samples.empty:
                y.append(0.0)
            else:
                y.append(len(df_samples) / len(df))

        # Plot the CDF
        sns.lineplot(y=y, x=x, ax=ax, color=next(colors), label=label)

    base.add_escape_velocity(ax, x_max, max(y))

    plt.legend()
    base.clear_default_labels(ax)
    base.savefig_and_close(filename, output_dir, close, fmt)
    return (fig, ax)


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
        xlabel="Effective Spin $\\chi_{eff}$",
        ylabel="PDF",
    )
    fig, ax = base.initialize_plot(figsize=(10, 8), labels=labels, padding=padding, fontsize=15)
    colors = iter(mcolors.TABLEAU_COLORS.keys())

    for label, df in dfs.items():
        _plot_pdf(ax, next(colors), df[S.EFF(C.SPIN)], label)

    plt.legend()
    base.clear_default_labels(ax)
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
        xlabel="Precession Spin $\\chi_p$",
        ylabel="PDF",
    )
    fig, ax = base.initialize_plot(figsize=(10, 8), labels=labels, padding=padding, fontsize=15)
    colors = iter(mcolors.TABLEAU_COLORS.keys())

    for label, df in dfs.items():
        _plot_pdf(ax, next(colors), df[S.PREC(C.SPIN)], label)

    plt.legend()
    base.clear_default_labels(ax)
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

    _series = series.dropna()
    density, bins = np.histogram(a=_series, bins=70, density=True)
    low, mid, high = _series.quantile(0.05), _series.quantile(0.5), _series.quantile(0.95)
    label = "%s: $%.2f_{-%.2f}^{+%.2f}$" % (name, mid, mid - low, high - mid)
    if unit:
        label += f" {unit}"
    ax.stairs(density, bins, label=label, color=color)


def table_estimates(
    dfs: dict[str, pd.DataFrame],
    filename="table_estimates",
    output_dir: Optional[str] = None,
    fmt: str = "md",
) -> pd.DataFrame:
    """Plot the posterior mass estimates.

    Args:
        dfs (dict[str, pd.DataFrame]): Key: name of the posterior, value: posterior dataframe.
        filename (str): Output filename.
        output_dir (Optional[str]): Output directory.
        close (bool): Whether to close the figure.
        fmt (str): The format of the visualizations. Defaults to "md".

    Returns:
        fig (plt.Figure): Figure.
        axes (plt.Axes): Axes.
    """

    col_to_names = {
        S.PRIMARY(C.MASS): "$m_1$",
        S.SECONDARY(C.MASS): "$m_2$",
        C.MASS_RATIO: "$q$",
        S.FINAL(C.MASS): "$m_f$",
        S.FINAL(C.SPIN_MAG): "$\\chi_f$",
        C.KICK: "$v_f$",
        S.PREC(C.SPIN): "$\\chi_{p}$",
        S.EFF(C.SPIN): "$\\chi_{eff}$",
    }
    data = {
        "": dfs.keys(),
        "Recovery Rate": [df[C.RECOVERY_RATE].iloc[0] for df in dfs.values()],
        **{f"p2g_{v_esc.short()}": [v_esc.compute_p2g(df) for df in dfs.values()] for v_esc in EscapeVelocity},
    }

    for col, name in col_to_names.items():
        data[name] = []
        for df in dfs.values():
            df = filter_unmapped_samples(df)
            low, mid, high = (
                df[col].quantile(0.05),
                df[col].quantile(0.5),
                df[col].quantile(0.95),
            )
            value = "$%.2f_{-%.2f}^{+%.2f}$" % (mid, mid - low, high - mid)
            data[name].append(value)

    df_table = pd.DataFrame(data)

    if output_dir:
        if fmt.lower() == "md":
            df_table.to_markdown(f"{output_dir}/{filename}.{fmt}", index=False)

        elif fmt.lower() == "csv":
            df_table.to_csv(f"{output_dir}/{filename}.{fmt}", index=False)

        else:
            local_logger.warning("Unsupported format: %s.", fmt)

    return df_table
