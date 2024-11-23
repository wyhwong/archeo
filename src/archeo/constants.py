import enum

import archeo.logger


local_logger = archeo.logger.get_logger(__name__)

SPEED_OF_LIGHT = 299792.458  # km/s


class Fits(enum.StrEnum):
    """Surrogate models for binary black hole merger simulations.

    Attributes:
        NRSur3dq8Remnant: non precessing BHs with mass ratio<=8, anti-/aligned spin <= 0.8.
        NRSur7dq4Remnant: precessing BHs with mass ratio<=4, generic spin <= 0.8.
        surfinBH7dq2: precessing BHs with mass ratio <= 2, generic spin <= 0.8.

    Details please refer to https://pypi.org/project/surfinBH/.
    """

    NRSUR3DQ8REMNANT = "NRSur3dq8Remnant"
    NRSUR7DQ4REMNANT = "NRSur7dq4Remnant"
    SURFINBH7DQ2 = "surfinBH7dq2"

    def load(self):
        """Load a surfinBH fits.

        Returns:
            fits (surfinBH.surfinBH.SurFinBH): The loaded fits.
        """

        import surfinBH

        local_logger.info(
            "Loading surfinBH %s, description: %s.",
            self,
            surfinBH.fits_collection[self].desc,
        )
        return surfinBH.LoadFits(self)


class EscapeVelocity(enum.Enum):
    """Escape velocity (Unit in km s^-1)"""

    GLOBULAR_CLUSTER = 50.0
    MILKY_WAY = 600.0
    NUCLEAR_STAR_CLUSTER = 1500.0
    ELLIPTICAL_GALAXY = 2500.0

    _LABELS = {
        "GLOBULAR_CLUSTER": "$v_{esc, GC}$",
        "MILKY_WAY": "$v_{esc, MW}$",
        "NUCLEAR_STAR_CLUSTER": "$v_{esc, NSC}$",
        "ELLIPTICAL_GALAXY": "$v_{esc, EG}$",
    }

    def label(self):
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

    @classmethod
    def to_vlines(cls) -> dict[str, float]:
        """Return a dictionary for vlines plotting

        Returns:
            vlines (Dict[str, float]): The escape velocity vlines.
        """

        return {esc_vel.label(): esc_vel.value for esc_vel in cls}


class Columns(enum.StrEnum):
    """Columns in the prior dataframe"""

    HEAVIER_BH_MASS = "m1"
    HEAVIER_BH_SPIN = "a1"
    LIGHTER_BH_MASS = "m2"
    LIGHTER_BH_SPIN = "a2"
    MASS_RATIO = "q"
    RETAINED_MASS = "m_ret"
    LIKELIHOOD = "l"
    RECOVERY_RATE = "r_rec"
    BH_MASS = "m_f"
    BH_KICK = "v_f"
    BH_SPIN = "a_f"
    BH_EFF_SPIN = "a_eff"
    BH_PREC_SPIN = "a_prec"
