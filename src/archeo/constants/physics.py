import enum

import pandas as pd


SPEED_OF_LIGHT = 299792.458  # km/s


class EscapeVelocity(enum.Enum):
    """Escape velocity (Unit in km s^-1)"""

    GLOBULAR_CLUSTER = 50.0
    MILKY_WAY = 600.0
    NUCLEAR_STAR_CLUSTER = 1500.0
    ELLIPTICAL_GALAXY = 2500.0

    def label(self) -> str:
        """Return the escape velocity label"""

        if self is EscapeVelocity.GLOBULAR_CLUSTER:
            return "$v_{esc, GC}$"

        if self is EscapeVelocity.MILKY_WAY:
            return "$v_{esc, MW}$"

        if self is EscapeVelocity.NUCLEAR_STAR_CLUSTER:
            return "$v_{esc, NSC}$"

        if self is EscapeVelocity.ELLIPTICAL_GALAXY:
            return "$v_{esc, EG}$"

        raise ValueError(f"Unknown escape velocity {self}")

    def short(self) -> str:
        """Return the short name of the escape velocity"""

        if self is EscapeVelocity.GLOBULAR_CLUSTER:
            return "GC"

        if self is EscapeVelocity.MILKY_WAY:
            return "MW"

        if self is EscapeVelocity.NUCLEAR_STAR_CLUSTER:
            return "NSC"

        if self is EscapeVelocity.ELLIPTICAL_GALAXY:
            return "EG"

        raise ValueError(f"Unknown escape velocity {self}")

    def compute_p2g(self, df: pd.DataFrame) -> float:
        """Return the probability of
        the black hole being a 2nd generation black hole
        under different escape velocity conditions."""

        if df.empty:
            return 0.0

        mask = (df["k_f"] <= self.value) & (df["m_1"] <= 65) & (df["m_2"] <= 65)
        return mask.sum() / len(df) * 100.0

    @classmethod
    def to_vlines(cls) -> dict[str, float]:
        """Return a dictionary for vlines plotting

        Returns:
            vlines (Dict[str, float]): The escape velocity vlines.
        """

        return {esc_vel.label(): esc_vel.value for esc_vel in cls}
