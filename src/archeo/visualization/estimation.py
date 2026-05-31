from typing import Optional

import corner
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from archeo.constants.physics import PISN_LB, PISN_UB, TypicalHostEscapeVelocity
from archeo.data_structures.visualization import Labels, Padding
from archeo.utils.logger import get_logger
from archeo.visualization import base


LOGGER = get_logger(__name__)


def filter_unmapped_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that are not mapped to posterior quantities.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Filtered dataframe with non-null kick values.
    """

    return df.dropna(subset=["k_f"])


def mass_estimates(
    df: pd.DataFrame,
    label: str,
    filename="mass_estimates",
    output_dir: Optional[str] = None,
    close: bool = True,
    fmt: str = "png",
):
    """Plot PDFs for component and remnant mass estimates.

    Args:
        df (pd.DataFrame): Posterior dataframe.
        label (str): Label for remnant distribution.
        filename (str): Output filename stem.
        output_dir (Optional[str]): Output directory.
        close (bool): Whether to close figure after saving.
        fmt (str): Figure format.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axis.
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
        "m_f": label + ": ",
        "m_1": "Heavier Parent: ",
        "m_2": "Lighter Parent: ",
    }
    for col, name in col_to_name.items():
        base.plot_pdf(ax, df[col], next(colors), name, unit=r"[$M_{\odot}$]")
    _add_pisn_gap(ax, next(colors))

    plt.legend()
    base.clear_default_labels(ax)
    base.savefig_and_close(filename, output_dir, close, fmt)
    return (fig, ax)


def _add_pisn_gap(ax, color: str) -> None:
    """Annotate pair-instability supernova mass gap boundaries on an axis.

    Args:
        ax: Matplotlib axis.
        color (str): Line color.

    Returns:
        None
    """

    ax.axvline(PISN_LB, color=color, linewidth=0.9, linestyle="--", label="PISN Gap")
    ax.axvline(PISN_UB, color=color, linewidth=0.9, linestyle="--")


def corner_estimates(  # pylint: disable=dangerous-default-value
    dfs: dict[str, pd.DataFrame],
    levels: list[float] = [0.68, 0.9],
    nbins: int = 70,
    filename="corner_estimates",
    output_dir: Optional[str] = None,
    close: bool = True,
    fmt: str = "png",
):
    """Generate corner plots for one or more posterior dataframes.

    Args:
        dfs (dict[str, pd.DataFrame]): Mapping from label to dataframe.
        levels (list[float]): Contour credibility levels.
        nbins (int): Histogram bin count.
        filename (str): Output filename stem.
        output_dir (Optional[str]): Output directory.
        close (bool): Whether to close figure after saving.
        fmt (str): Figure format.

    Returns:
        tuple[plt.Figure, plt.Axes]: Last generated figure and axes.
    """

    corner_type_to_var_names = {
        "part": ["m_1", "m_2", "k_f"],
        "full": [
            "m_1",
            "m_2",
            "m_f",
            "k_f",
            "a_f",
            "chi_eff",
        ],
    }
    corner_type_to_labels = {
        "part": [
            r"$m_1$ [$M_{\odot}$]",
            r"$m_2$ [$M_{\odot}$]",
            r"$v_f$ [km s$^{-1}$]",
        ],
        "full": [
            r"$m_1$ [$M_{\odot}$]",
            r"$m_2$ [$M_{\odot}$]",
            r"$m_f$ [$M_{\odot}$]",
            r"$v_f$ [km s$^{-1}$]",
            "$a_f$",
            "$\\chi_{eff}$",
        ],
    }

    # Add precession spin if available
    prec_spins = [df["chi_p"].max() for df in dfs.values()]
    if max(prec_spins) > 0.0:
        if np.isclose(min(prec_spins), 0.0):
            LOGGER.warning("Precession spin is not available for all dataframes, " "skipped adding it to corner plot.")
        else:
            corner_type_to_var_names["full"].append("chi_p")
            corner_type_to_labels["full"].append("$\\chi_{p}$")

    for corner_type, var_names in corner_type_to_var_names.items():
        fig, axes = base.initialize_plot(ncols=len(var_names), nrows=len(var_names), figsize=(9, 9))
        colors = iter(mcolors.TABLEAU_COLORS.keys())
        handles = []

        for label, df in dfs.items():
            df = filter_unmapped_samples(df)

            if len(df) < nbins:
                LOGGER.info("Dataframe does not have enough samples to plot.")
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
    """Plot second-generation probability as a function of escape velocity.

    Args:
        dfs (dict[str, pd.DataFrame]): Mapping from label to dataframe.
        filename (str): Output filename stem.
        output_dir (Optional[str]): Output directory.
        close (bool): Whether to close figure after saving.
        fmt (str): Figure format.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axis.
    """

    # Set up x-axis
    x_max = max(df["k_f"].max() for df in dfs.values())
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
            df_samples = df.loc[(df["k_f"] <= kick) & (df["m_1"] <= PISN_LB) & (df["m_2"] <= PISN_LB)]
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
    """Plot effective-spin PDFs for labeled posterior sets.

    Args:
        dfs (dict[str, pd.DataFrame]): Mapping from label to dataframe.
        filename (str): Output filename stem.
        output_dir (Optional[str]): Output directory.
        close (bool): Whether to close figure after saving.
        fmt (str): Figure format.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axis.
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
        base.plot_pdf(ax, df["chi_eff"], next(colors), label)

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
    """Plot precession-spin PDFs for labeled posterior sets.

    Args:
        dfs (dict[str, pd.DataFrame]): Mapping from label to dataframe.
        filename (str): Output filename stem.
        output_dir (Optional[str]): Output directory.
        close (bool): Whether to close figure after saving.
        fmt (str): Figure format.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axis.
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
        base.plot_pdf(ax, df["chi_p"], next(colors), label)

    plt.legend()
    base.clear_default_labels(ax)
    base.savefig_and_close(filename, output_dir, close, fmt)
    return (fig, ax)


def table_estimates(
    dfs: dict[str, pd.DataFrame],
    filename="table_estimates",
    output_dir: Optional[str] = None,
    fmt: str = "md",
) -> pd.DataFrame:
    """Build and optionally export posterior summary table.

    Args:
        dfs (dict[str, pd.DataFrame]): Mapping from label to dataframe.
        filename (str): Output filename stem.
        output_dir (Optional[str]): Output directory for export.
        fmt (str): Export format, currently `md` or `csv`.

    Returns:
        pd.DataFrame: Summary table dataframe.
    """

    col_to_names = {
        "m_1": "$m_1$",
        "m_2": "$m_2$",
        "q": "$q$",
        "m_f": "$m_f$",
        "a_f": "$a_f$",
        "k_f": "$v_f$",
        "chi_p": "$\\chi_{p}$",
        "chi_eff": "$\\chi_{eff}$",
    }
    data = {
        "": dfs.keys(),
        "Recovery Rate": [df["m_1"].notna().sum() / df.shape[0] for df in dfs.values()],
        **{f"p2g_{v_esc.short}": [v_esc.compute_p2g(df) for df in dfs.values()] for v_esc in TypicalHostEscapeVelocity},
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
            LOGGER.warning("Unsupported format: %s.", fmt)

    return df_table
