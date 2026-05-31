from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation

from archeo.utils.logger import get_logger


logger = get_logger(__name__)


def animate_remnant_property_change_over_kick(
    df: pd.DataFrame,
    col_name: str,
    kick_lb=50.0,
    kick_width=50.0,
    output_dir: Optional[str] = None,
):
    """Animate evolution of a remnant-property distribution versus kick threshold.

    Args:
        df (pd.DataFrame): Input dataframe.
        col_name (str): Column to animate.
        kick_lb (float): Initial kick threshold.
        kick_width (float): Step size between thresholds.
        output_dir (Optional[str]): Optional output directory for GIF export.

    Returns:
        Optional[FuncAnimation]: Animation object, or `None` if configuration is invalid.
    """

    _labels = {
        "m_f": "Remnant Mass",
        "a_f": "Remnant Spin",
        "q": "Mass Ratio",
        "chi_eff": "Effective Spin",
        "chi_p": "Precession Spin",
    }

    if col_name not in _labels:
        logger.warning("Parameter %s is not supported for animation. ", col_name)
        return None

    label = _labels[col_name]

    kick_ub = df["k_f"].max()
    if kick_lb >= kick_ub:
        logger.warning(
            "Kick lower bound %.2f is greater than or equal to upper bound %.2f. " "No animation will be created.",
            kick_lb,
            kick_ub,
        )
        return None

    nbins = int((kick_ub - kick_lb) / kick_width) + 1
    k_bounds = np.linspace(kick_lb, kick_ub, nbins)
    binwidth = (df[col_name].max() - df[col_name].min()) / 70

    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame: int) -> plt.Axes:
        """Render one animation frame for a specific kick-threshold index.

        Args:
            frame (int): Frame index.

        Returns:
            matplotlib.axes.Axes: Updated axis.
        """

        ax.clear()
        values = df.loc[df["k_f"] <= k_bounds[frame], col_name]

        if values.nunique() == 1:
            ax.axvline(values.iloc[0], color="red", linestyle="--", linewidth=2)
            ax.set_title(f"{label} (Kick $v_f$ <= {k_bounds[frame]:.2f})")
            ax.set_xlabel(label)
            ax.set_ylabel("Density")
            return ax

        sns.histplot(values, ax=ax, stat="density", binwidth=binwidth, fill=False, element="step")
        ax.set(xlabel=label, ylabel="Density")
        ax.grid(True)
        ax.set_title(f"Distribution of {label} (Kick $v_f$ <= {k_bounds[frame]:.2f})")
        return ax

    ani = FuncAnimation(fig, update, frames=len(k_bounds), interval=100)
    if output_dir:
        filename = label.lower().replace(" ", "_")
        ani.save(f"{output_dir}/{filename}.gif", writer="pillow")

    return ani
