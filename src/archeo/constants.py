import enum
import archeo.logger

local_logger = archeo.logger.get_logger(__name__)

SPEED_OF_LIGHT = 299792.458  # km/s


class Fits(enum.StrEnum):
    """
    Surrogate models.

    NRSur3dq8Remnant:
        non precessing BHs with mass ratio<=8, anti-/aligned spin <= 0.8

    NRSur7dq4Remnant:
        precessing BHs with mass ratio<=4, generic spin <= 0.8

    surfinBH7dq2:
        precessing BHs with mass ratio <= 2, generic spin <= 0.8
    """

    NRSUR3DQ8REMNANT = "NRSur3dq8Remnant"
    NRSUR7DQ4REMNANT = "NRSur7dq4Remnant"
    SURFINBH7DQ2 = "surfinBH7dq2"

    def load(self):
        """
        Load a surfinBH fits.

        Returns:
        -----
            fits (surfinBH.surfinBH.SurFinBH):
                The loaded fits.
        """

        import surfinBH

        local_logger.info(
            "Loading surfinBH %s, description: %s.",
            self,
            surfinBH.fits_collection[self].desc,
        )
        return surfinBH.LoadFits(self)


class EscapeVelocityLabel(enum.Enum):
    """Label for escape velocity (used in visualization labels)"""

    GLOBULAR_CLUSTER = "$v_{esc, GC}$"
    MILKY_WAY = "$v_{esc, MW}$"
    NUCLEAR_STAR_CLUSTER = "$v_{esc, NSC}$"
    ELLIPTICAL_GALAXY = "$v_{esc, EG}$"


class EscapeVelocity(enum.Enum):
    """Escape velocity (Unit in km s^-1)"""

    GLOBULAR_CLUSTER = 50.0
    MILKY_WAY = 600.0
    NUCLEAR_STAR_CLUSTER = 1500.0
    ELLIPTICAL_GALAXY = 2500.0

    @classmethod
    def to_vlines(cls):
        """Return a dictionary for vlines plotting

        The key is the escape velocity label and the value is the escape velocity.
        """

        return {
            EscapeVelocityLabel.GLOBULAR_CLUSTER: cls.GLOBULAR_CLUSTER.value,
            EscapeVelocityLabel.MILKY_WAY: cls.MILKY_WAY.value,
            EscapeVelocityLabel.NUCLEAR_STAR_CLUSTER: cls.NUCLEAR_STAR_CLUSTER.value,
            EscapeVelocityLabel.ELLIPTICAL_GALAXY: cls.ELLIPTICAL_GALAXY.value,
        }


class Columns(enum.StrEnum):
    """Columns in the prior dataframe"""

    HEAVIER_BH_MASS = "m1"
    LIGHTER_BH_MASS = "m2"
    MASS_RATIO = "q"
    REMNANT_BH_MASS = "mf"
    REMNANT_RECOIL = "vf"
    RETAINED_PORTION = "rp"
    SPIN_MAGNITUDE = "chif"
    LIKELIHOOD = "likelihood"
    RECOVERYRATE = "recovery_rate"
    EFFECTIVE_SPIN = "a_eff"
    PRECESSION_SPIN = "a_prec"
