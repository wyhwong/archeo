import enum
import os

import pandas as pd
from pydantic import BaseModel, NonNegativeFloat


SPEED_OF_LIGHT = 299792.458  # km/s
BH_MASS_LB: float = float(os.environ.get("BH_MASS_LB", 5.0))  # Solar masses
PISN_LB: float = float(os.environ.get("PISN_LB", 65.0))  # Solar masses
PISN_UB: float = float(os.environ.get("PISN_UB", 130.0))  # Solar masses
BH_SPIN_UB: float = float(os.environ.get("BH_SPIN_UB", 0.99))  # Dimensionless spin


class _TypicalHostEscapeVelocityMeta(BaseModel):
    v_esc: NonNegativeFloat  # km s^-1
    short: str
    latex: str


class TypicalHostEscapeVelocity(enum.Enum):
    """Escape velocity (units: km s^-1)."""

    GLOBULAR_CLUSTER = _TypicalHostEscapeVelocityMeta(v_esc=50.0, short="GC", latex=r"$v_{esc, GC}$")
    MILKY_WAY = _TypicalHostEscapeVelocityMeta(v_esc=600.0, short="MW", latex=r"$v_{esc, MW}$")
    NUCLEAR_STAR_CLUSTER = _TypicalHostEscapeVelocityMeta(v_esc=1500.0, short="NSC", latex=r"$v_{esc, NSC}$")
    ELLIPTICAL_GALAXY = _TypicalHostEscapeVelocityMeta(v_esc=2500.0, short="EG", latex=r"$v_{esc, EG}$")

    @property
    def v_esc(self) -> float:
        """Return host escape velocity.

        Returns:
            float: Escape velocity in km/s.
        """

        return self.value.v_esc

    @property
    def short(self) -> str:
        """Return short host label.

        Returns:
            str: Abbreviated host name.
        """

        return self.value.short

    @property
    def latex(self) -> str:
        """Return display label string.

        Returns:
            str: Label representation.
        """

        return self.value.latex

    def compute_p2g(
        self,
        df: pd.DataFrame,
        kf_col: str = "k_f",
        m1_col: str = "m_1",
        m2_col: str = "m_2",
    ) -> float:
        """Compute second-generation percentage under this host escape velocity.

        Args:
            df (pd.DataFrame): Sample dataframe containing kick and component masses.
            kf_col (str): Kick-velocity column name.
            m1_col (str): Primary-mass column name.
            m2_col (str): Secondary-mass column name.

        Returns:
            float: Percentage of rows satisfying 2G criteria.
        """

        if df.empty:
            return 0.0

        mask = (df[kf_col] <= self.v_esc) & (df[m1_col] <= PISN_LB) & (df[m2_col] <= PISN_LB)
        return mask.mean() * 100.0

    @classmethod
    def latex_to_values(cls) -> dict[str, float]:
        """Map display labels to escape-velocity values.

        Returns:
            dict[str, float]: Label-to-velocity mapping.
        """

        return {m.latex: m.v_esc for m in cls}

    @classmethod
    def short_to_values(cls) -> dict[str, float]:
        """Map short host labels to escape-velocity values.

        Returns:
            dict[str, float]: Abbreviation-to-velocity mapping.
        """

        return {m.short: m.v_esc for m in cls}
