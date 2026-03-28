import enum

import pandas as pd
from pydantic import BaseModel, NonNegativeFloat


SPEED_OF_LIGHT = 299792.458  # km/s


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
        return self.value.v_esc

    @property
    def short(self) -> str:
        return self.value.short

    @property
    def latex(self) -> str:
        return self.value.latex

    def compute_p2g(
        self,
        df: pd.DataFrame,
        max_mass: float = 65.0,
        kf_col: str = "k_f",
        m1_col: str = "m_1",
        m2_col: str = "m_2",
    ) -> float:
        """Probability of being 2nd-generation under this escape velocity."""

        if df.empty:
            return 0.0

        mask = (df[kf_col] <= self.v_esc) & (df[m1_col] <= max_mass) & (df[m2_col] <= max_mass)
        return mask.mean() * 100.0

    @classmethod
    def latex_to_values(cls) -> dict[str, float]:
        return {m.latex: m.v_esc for m in cls}

    @classmethod
    def short_to_values(cls) -> dict[str, float]:
        return {m.short: m.v_esc for m in cls}
