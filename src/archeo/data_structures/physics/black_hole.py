from typing import TypeAlias, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, NonNegativeFloat, PositiveFloat, field_validator

from archeo.data_structures.annotation import Distribution
from archeo.data_structures.distribution import Uniform


class BlackHole(BaseModel, frozen=True):
    """Black hole data class"""

    mass: PositiveFloat
    spin_magnitude: PositiveFloat
    spin_vector: tuple[float, float, float]
    speed: NonNegativeFloat

    @property
    def horizontal_spin(self) -> float:
        """Calculate the horizontal spin component of the black hole."""

        return np.sqrt(self.spin_vector[0] ** 2 + self.spin_vector[1] ** 2)

    @property
    def vertical_spin(self) -> float:
        """Calculate the vertical spin component of the black hole."""

        return self.spin_vector[2]


BlackHoles: TypeAlias = list[BlackHole]


class BlackHoleGenerator(BaseModel, frozen=True):
    """Black hole generator data class."""

    mass_distribution: Distribution = Uniform(low=5, high=65)
    spin_magnitude_distribution: Distribution = Uniform(low=0, high=1)
    phi_distribution: Distribution = Uniform(low=0, high=2 * np.pi)
    theta_distribution: Distribution = Uniform(low=0, high=np.pi)

    @field_validator("spin_magnitude_distribution", mode="before")
    @classmethod
    def validate_spin_magnitude_distribution(cls, v):
        """Validate that the spin magnitude distribution is within the range [0, 1]."""

        if (v.min < 0) or (v.max > 1):
            raise ValueError("Spin magnitude distribution must be within the range [0, 1].")

        return v

    @field_validator("phi_distribution", mode="before")
    @classmethod
    def validate_phi_distribution(cls, v):
        """Validate that the phi distribution is within the range [0, 2 * pi]."""

        if (v.min < 0) or (v.max > 2 * np.pi):
            raise ValueError("Phi distribution must be within the range [0, 2 * pi].")

        return v

    @field_validator("theta_distribution", mode="before")
    @classmethod
    def validate_theta_distribution(cls, v):
        """Validate that the theta distribution is within the range [0, pi]."""

        if (v.min < 0) or (v.max > np.pi):
            raise ValueError("Theta distribution must be within the range [0, pi].")

        return v

    def draw(self, size: int = 1) -> BlackHoles:
        """Generate a list of black holes based on the specified distributions."""

        masses = self.mass_distribution.draw(size=size)
        spin_magnitudes = self.spin_magnitude_distribution.draw(size=size)
        phis = self.phi_distribution.draw(size=size)
        thetas = self.theta_distribution.draw(size=size)

        black_holes = []
        for mass, spin_magnitude, phi, theta in zip(masses, spin_magnitudes, phis, thetas):
            spin_vector = (
                spin_magnitude * np.sin(theta) * np.cos(phi),
                spin_magnitude * np.sin(theta) * np.sin(phi),
                spin_magnitude * np.cos(theta),
            )
            black_holes.append(BlackHole(mass=mass, spin_magnitude=spin_magnitude, spin_vector=spin_vector, speed=0.0))

        return black_holes


class BlackHolePopulation(BaseModel, frozen=True):
    """Black hole population data class."""

    black_holes: BlackHoles

    def draw(self, size: int = 1) -> BlackHoles:
        """Draw a sample of black holes from the population."""

        return np.random.choice(self.black_holes, size=size, replace=True).tolist()

    @classmethod
    def from_simulation_results(
        cls,
        df: pd.DataFrame,
        phi_distribution: Distribution = Uniform(low=0, high=2 * np.pi),
        theta_distribution: Distribution = Uniform(low=0, high=np.pi),
    ) -> "BlackHolePopulation":
        """Create a black hole population from simulation results."""

        phis = phi_distribution.draw(size=len(df))
        thetas = theta_distribution.draw(size=len(df))

        return cls(
            black_holes=[
                BlackHole(
                    mass=merger.m_f,
                    spin_magnitude=merger.a_f,
                    spin_vector=(
                        merger.a_f * np.sin(thetas[merger.Index]) * np.cos(phis[merger.Index]),
                        merger.a_f * np.sin(thetas[merger.Index]) * np.sin(phis[merger.Index]),
                        merger.a_f * np.cos(thetas[merger.Index]),
                    ),
                    speed=merger.k_f,
                )
                for merger in df.reset_index(drop=True).itertuples()
            ]
        )


BlackHoleSource: TypeAlias = Union[BlackHoleGenerator, BlackHolePopulation]
