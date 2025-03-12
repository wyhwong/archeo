import json
from dataclasses import asdict, dataclass

import numpy as np
import yaml

from archeo.constants import Fits


@dataclass(frozen=True)
class Domain:
    """Domain of a parameter.

    Attributes:
        low (float): Lower bound of the domain
        high (float): Upper bound of the domain
    """

    low: float = float("-inf")
    high: float = float("inf")

    def contain(self, value: float) -> bool:
        """Check if the input value is within the domain."""

        return self.low <= value <= self.high

    def draw(self) -> float:
        """Draw a random value from the domain."""

        return np.random.uniform(low=self.low, high=self.high)


@dataclass(frozen=True)
class Binary:
    """Binary parameters.

    Attributes:
        m_1 (float): Mass of the primary black hole (in solar mass)
        m_2 (float): Mass of the secondary black hole (in solar mass)
        chi_1 (tuple[float, float, float]): Spin of the primary black hole (dimensionless), [0, 1]
        chi_2 (tuple[float, float, float]): Spin of the secondary black hole (dimensionless), [0, 1]
    """

    m_1: float
    m_2: float
    chi_1: tuple[float, float, float]
    chi_2: tuple[float, float, float]


@dataclass(frozen=True)
class Event:
    """Event (Binary Black Hole Merger) parameters."""

    m_1: float
    m_2: float
    m_ret: float
    m_ret_err: float
    v_f: tuple[float, float, float]
    v_f_err: tuple[float, float, float]
    chi_1: tuple[float, float, float]
    chi_2: tuple[float, float, float]
    chi_f: tuple[float, float, float]
    chi_f_err: tuple[float, float, float]


@dataclass(frozen=True)
class PriorConfig:
    """Configuration of the prior.

    Attributes:
        n_samples (int): Number of samples to generate.
        fits (Fits): Surrogate model to use.
        is_spin_aligned (bool): Whether the spins are aligned or not.
        is_only_up_aligned_spin (bool): Whether the spins are only in the positive z-direction.
        spin (Domain): Domain of the spin parameter.
        phi (Domain): Domain of the azimuthal angle of the spin.
        theta (Domain): Domain of the polar angle of the spin.
        mass_ratio (Domain): Domain of the mass ratio.
        mass (Domain): Domain of the mass.
        is_mahapatra (bool): Whether the Mahapatra mass function is used.
    """

    n_samples: int
    fits: Fits
    is_spin_aligned: bool
    is_only_up_aligned_spin: bool
    a_1: Domain  # unit: dimensionless
    a_2: Domain  # unit: dimensionless
    phi_1: Domain  # unit: pi
    phi_2: Domain  # unit: pi
    theta_1: Domain  # unit: pi
    theta_2: Domain  # unit: pi
    mass_ratio: Domain  # unit: dimensionless
    m_1: Domain  # unit: solar mass
    m_2: Domain  # unit: solar mass
    is_mahapatra: bool = False
    is_uniform_in_mass_ratio: bool = False

    def __post_init__(self) -> None:
        """Post initialization."""

        self._check_fits_is_valid()
        self._check_is_aligned_spin_if_only_up_aligned_spin()
        self._check_is_m1_highbound_greater_than_m2_lowbound()

    def _check_fits_is_valid(self) -> None:
        """Check if the surrogate model is valid."""

        if self.fits not in Fits:
            raise ValueError(f"Invalid fits: {self.fits}")

    def _check_is_aligned_spin_if_only_up_aligned_spin(self) -> None:
        """Check if spins are aligned if only up-aligned spin is set."""

        if self.is_only_up_aligned_spin and not self.is_spin_aligned:
            raise ValueError("Only up-aligned spin is only valid when spins are aligned.")

    def _check_is_m1_highbound_greater_than_m2_lowbound(self) -> None:
        """Check if the high bound of m_1 is greater than the low bound of m_2."""

        if self.m_1.high <= self.m_2.low:
            raise ValueError("The high bound of m_1 must be greater than the low bound of m_2.")

    @staticmethod
    def from_dict(data: dict) -> "PriorConfig":
        """Create a PriorConfig object from a dictionary."""

        return PriorConfig(
            n_samples=data["n_samples"],
            fits=Fits(data["fits"]),
            is_spin_aligned=data["is_spin_aligned"],
            is_only_up_aligned_spin=data["is_only_up_aligned_spin"],
            a_1=Domain(data["a_1"]["low"], data["a_1"]["high"]),
            a_2=Domain(data["a_2"]["low"], data["a_2"]["high"]),
            phi_1=Domain(data["phi_1"]["low"], data["phi_1"]["high"]),
            phi_2=Domain(data["phi_2"]["low"], data["phi_2"]["high"]),
            theta_1=Domain(data["theta_1"]["low"], data["theta_1"]["high"]),
            theta_2=Domain(data["theta_2"]["low"], data["theta_2"]["high"]),
            mass_ratio=Domain(data["mass_ratio"]["low"], data["mass_ratio"]["high"]),
            m_1=Domain(data["m_1"]["low"], data["m_1"]["high"]),
            m_2=Domain(data["m_2"]["low"], data["m_2"]["high"]),
            is_mahapatra=data["is_mahapatra"],
            is_uniform_in_mass_ratio=data["is_uniform_in_mass_ratio"],
        )

    @classmethod
    def from_json(cls, filepath: str) -> "PriorConfig":
        """Create a PriorConfig object from a JSON file."""

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, filepath: str) -> "PriorConfig":
        """Create a PriorConfig object from a YAML file."""

        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Convert to a dictionary."""

        return asdict(self)

    def to_json(self, filepath: str) -> None:
        """Save to a JSON file."""

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    def to_yaml(self, filepath: str) -> None:
        """Save to a YAML file."""

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f)


@dataclass(frozen=True)
class Padding:
    """
    Padding for plot.

    Attributes:
        tpad: float, the top padding of the plot.
        lpad: float, the left padding of the plot.
        bpad: float, the bottom padding of the plot.
    """

    tpad: float = 2.5
    lpad: float = 0.1
    bpad: float = 0.12


@dataclass(frozen=True)
class Labels:
    """
    Labels for plot.

    Attributes:
        title: str, the title of the plot.
        xlabel: str, the x-axis label of the plot.
        ylabel: str, the y-axis label of the plot.
    """

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
