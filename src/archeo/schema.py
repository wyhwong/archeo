from dataclasses import dataclass

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

    low = float("-inf")
    high = float("inf")

    def check(self, value: float) -> bool:
        """Check if the input value is within the domain."""

        return self.low <= value <= self.high

    def draw(self) -> float:
        """Draw a random value from the domain."""

        return np.random.uniform(low=self.low, high=self.high)


@dataclass(frozen=True)
class Binary:
    """Binary parameters.

    NOTE:
    - m1 and m2 are optional,
        we can rescale them anytime after simulation.
        Empirically tested, prior with masses injected,
        and not injected lead to the same result.

    Attributes:
        m1 (float): Mass of the primary black hole (in solar mass)
        m2 (float): Mass of the secondary black hole (in solar mass)
        chi1 (tuple[float, float, float]): Spin of the primary black hole (dimensionless), [0, 1]
        chi2 (tuple[float, float, float]): Spin of the secondary black hole (dimensionless), [0, 1]
    """

    m1: float
    m2: float
    chi1: tuple[float, float, float]
    chi2: tuple[float, float, float]


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
    spin: Domain
    phi: Domain
    theta: Domain
    mass_ratio: Domain
    mass: Domain
    is_mahapatra: bool

    def __post_init__(self) -> None:
        """Post initialization."""

        if self.fits not in Fits:
            raise ValueError(f"Invalid fits: {self.fits}")

        if self.is_only_up_aligned_spin and not self.is_spin_aligned:
            raise ValueError("Only up-aligned spin is only valid when spins are aligned.")

    def from_yaml(self, filepath: str) -> "PriorConfig":
        """Load prior configuration from a yaml file.

        Args:
            filepath (str): Path to the yaml file.

        Returns:
            PriorConfig: Prior configuration.
        """

        with open(filepath, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return PriorConfig(**config)


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
