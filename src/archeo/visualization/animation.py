from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation

from archeo.constants import Columns as C
from archeo.constants import Suffixes as S
from archeo.logger import get_logger


logger = get_logger(__name__)


def animate_remnant_property_change_over_kick(
    df: pd.DataFrame,
    parameter: Literal[C.MASS, C.SPIN_MAG],
    kick_lb=50.0,
    nbins=50,
    output_dir: Optional[str] = None,
):
    """Create an animation of the remnant mass/spin distribution change over different kick values

    Args:
        df (pd.DataFrame): DataFrame containing the prior data.
        parameter (C): The remnant property to animate, either C.MASS or C.S
        kick_lb (float): Lower bound of the kick velocity to consider for the animation.
        nbins (int): Number of bins for the histogram.
        output_dir (Optional[str]): Directory to save the animation. If None, the animation will
            not be saved to a file.

    Returns:
        ani (FuncAnimation): The animation object.
    """

    if parameter not in [C.MASS, C.SPIN_MAG]:
        logger.warning(
            "Parameter %s is not supported for animation. " "Only 'mass' and 'spin' are valid parameters.",
            parameter,
        )
    label = r"Remnant Mass [$M_\odot$]" if parameter == C.MASS else "Remnant Spin"

    kick_ub = df[C.KICK].max()
    if kick_lb >= kick_ub:
        logger.warning(
            "Kick lower bound %.2f is greater than or equal to upper bound %.2f. " "No animation will be created.",
            kick_lb,
            kick_ub,
        )
        return None

    k_bounds = np.linspace(kick_lb, kick_ub, nbins)
    col = S.FINAL(parameter)
    binwidth = (df[col].max() - df[col].min()) / nbins

    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        ax.clear()
        sns.histplot(
            df.loc[df[C.KICK] <= k_bounds[frame], col],
            ax=ax,
            stat="density",
            binwidth=binwidth,
            fill=False,
            element="step",
        )
        ax.set(xlabel=label, ylabel="Density")
        ax.grid(True)
        ax.set_title(f"Distribution of {label} (Kick $v_f$ <= {k_bounds[frame]:.2f})")
        return ax

    ani = FuncAnimation(fig, update, frames=len(k_bounds), interval=100)
    if output_dir:
        filename = "mass" if parameter == C.MASS else "spin"
        ani.save(f"{output_dir}/{filename}.gif", writer="pillow")

    return ani
