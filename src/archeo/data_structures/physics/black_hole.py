from typing import Optional, TypeAlias

import numpy as np
from pydantic import BaseModel, PositiveFloat, field_validator

from archeo.data_structures.distribution import DistributionBase, Uniform


class BlackHole(BaseModel, frozen=True):
    """Black hole data class"""

    mass: PositiveFloat
    spin_magnitude: PositiveFloat
    spin_vector: tuple[float, float, float]
    speed: PositiveFloat


BlackHoles: TypeAlias = list[BlackHole]


class BlackHoleGenerator(BaseModel, frozen=True):
    """Black hole generator data class."""

    mass_distribution: DistributionBase = Uniform(low=5, high=65)
    spin_magnitude_distribution: DistributionBase = Uniform(low=0, high=1)
    phi_distribution: DistributionBase = Uniform(low=0, high=2 * np.pi)
    theta_distribution: DistributionBase = Uniform(low=0, high=np.pi)

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

    def generate(self, size: Optional[int] = None) -> BlackHoles:
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
