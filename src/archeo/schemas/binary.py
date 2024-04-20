import enum
from dataclasses import dataclass
from typing import Iterator, Optional

import archeo.schemas.common


SPEED_OF_LIGHT = 299792.458  # km/s


@dataclass
class Binary:
    """
    Binary parameters.

    NOTE: m1 and m2 are optional because we can rescale them anytime after simulation.
          Empirically tested, prior with masses injected and not injected lead to the same result.

    Attributes
    -----
    mass_ratio (float):
        Mass ratio of the binary (dimensionless)

    chi1 (tuple[float, float, float]):
        Spin of the primary black hole (dimensionless), [0, 1]

    chi2 (tuple[float, float, float]):
        Spin of the secondary black hole (dimensionless), [0, 1]

    m1 (float):
        Mass of the primary black hole (in solar mass)

    m2 (float):
        Mass of the secondary black hole (in solar mass)
    """

    mass_ratio: float
    chi1: tuple[float, float, float]
    chi2: tuple[float, float, float]
    m1: Optional[float] = None
    m2: Optional[float] = None


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


@dataclass
class BinarySettings:
    """
    Binary configuration.

    Attributes
    -----
    is_spin_aligned (bool):
        Whether the spins are aligned or not

    only_up_aligned_spin (bool):
        Whether the spins are only in the positive z-direction

    spin (archeo.schemas.common.Domain):
        Domain of the spin parameter

    phi (archeo.schemas.common.Domain):
        Domain of the azimuthal angle of the spin

    theta (archeo.schemas.common.Domain):
        Domain of the polar angle of the spin

    mass_ratio (archeo.schemas.common.Domain):
        Domain of the mass ratio

    mass (archeo.schemas.common.Domain):
        Domain of the mass

    is_mahapatra (bool):
        Whether the Mahapatra mass function is used
    """

    is_spin_aligned: bool
    only_up_aligned_spin: bool
    spin: archeo.schemas.common.Domain
    phi: archeo.schemas.common.Domain
    theta: archeo.schemas.common.Domain
    mass_ratio: archeo.schemas.common.Domain
    mass: archeo.schemas.common.Domain
    is_mahapatra: bool

    def __post_init__(self) -> None:
        """Post initialization."""

        if self.only_up_aligned_spin and not self.is_spin_aligned:
            raise ValueError("Only up-aligned spin is only valid when spins are aligned.")


class EscapeVelocity(enum.Enum):
    """Escape velocity (Unit in km s^-1)"""

    GLOBULAR_CLUSTER = 50.0
    MILKY_WAY = 600.0
    NUCLEAR_STAR_CLUSTER = 1500.0
    ELLIPTICAL_GALAXY = 2500.0

    @staticmethod
    def value_iter() -> Iterator[float]:
        """Get the escape velocity."""

        return iter([v.value for v in EscapeVelocity])

    @staticmethod
    def label_iter() -> Iterator[str]:
        """Get the escape velocity label (for visualization)."""

        return EscapeVelocityLabel.value_iter()


class EscapeVelocityLabel(enum.Enum):
    """Escape velocity (Unit in km s^-1)"""

    GLOBULAR_CLUSTER = "$v_{esc}$ Globular Cluster"
    MILKY_WAY = "$v_{esc}$ Milky Way"
    NUCLEAR_STAR_CLUSTER = "$v_{esc}$ Nuclear Star Cluster"
    ELLIPTICAL_GALAXY = "$v_{esc}$ (Elliptical Galaxy)"

    @staticmethod
    def value_iter() -> Iterator[float]:
        """Get the escape velocity label (for visualization)."""

        return iter([label.value for label in EscapeVelocityLabel])
