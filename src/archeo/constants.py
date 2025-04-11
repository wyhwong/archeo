import enum
import os

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

        import numpy as np  # pylint: disable=import-outside-toplevel
        import surfinBH  # pylint: disable=import-outside-toplevel

        # TODO: Remove this when SurfinBH upgraded to numpy 2.0
        np.string_ = np.bytes_  # Here we fix the numpy incompatibility issue in SurfinBH

        local_logger.info(
            "Loading surfinBH %s, description: %s.",
            self.value,
            surfinBH.fits_collection[self.value].desc,
        )

        try:
            return surfinBH.LoadFits(self.value)
        except (OSError, KeyError) as e:
            local_logger.error("Failed to load surfinBH %s: %s", self.value, str(e))
            self.clean_up_surfinbh_data()
            return self.load()

    @staticmethod
    def clean_up_surfinbh_data():
        """Clean up the surfinBH data directory.

        We clean up in two situations:
        1. KeyError: this seems to be a bug in surfinBH,
           when installing the latest version (1.2.6).
        2. OSError: this happens when we interrupt the download
           of the surfinBH data files.
        """

        import surfinBH  # pylint: disable=import-outside-toplevel

        # Remove all files in the data directory
        data_dir = f"{os.path.dirname(surfinBH.__file__)}/data"

        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                local_logger.warning(
                    "Cleaning up surfinBH data directory: removing %s due to error.",
                    f"{data_dir}/{file}",
                )
                os.remove(f"{data_dir}/{file}")


class EscapeVelocity(enum.Enum):
    """Escape velocity (Unit in km s^-1)"""

    GLOBULAR_CLUSTER = 50.0
    MILKY_WAY = 600.0
    NUCLEAR_STAR_CLUSTER = 1500.0
    ELLIPTICAL_GALAXY = 2500.0

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

    HEAVIER_BH_MASS = "m_1"
    HEAVIER_BH_SPIN = "a_1"
    HEAVIER_BH_CHI = "chi_1"
    LIGHTER_BH_MASS = "m_2"
    LIGHTER_BH_SPIN = "a_2"
    LIGHTER_BH_CHI = "chi_2"
    MASS_RATIO = "q"
    RETAINED_MASS = "m_ret"
    LIKELIHOOD = "l"
    RECOVERY_RATE = "r_rec"
    BH_MASS = "m_f"
    BH_KICK = "k_f"
    BH_VEL = "v_f"
    BH_CHI = "chi_f"
    BH_SPIN = "a_f"
    BH_EFF_SPIN = "chi_eff"
    BH_PREC_SPIN = "chi_p"


class Prefixes(enum.StrEnum):
    """Prefixes for columns"""

    ORIGINAL = "original"

    def __call__(self, column: Columns) -> str:
        """Return the column name with prefix"""

        return f"{self.value}_{column.value}"
