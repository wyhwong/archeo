from typing import Optional, TypeAlias, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, NonNegativeFloat, PositiveFloat, field_validator

from archeo.constants.physics import BH_MASS_LB, BH_SPIN_UB, PISN_LB
from archeo.data_structures.distribution import Uniform
from archeo.data_structures.type_alias import Distribution


class BlackHole(BaseModel, frozen=True):
    """Single black-hole state with mass and spin-vector components.

    Stores scalar mass and Cartesian spin components, and exposes convenient
    derived spin projections (orbital-plane and z-axis components).
    """

    mass: PositiveFloat
    spin_magnitude: NonNegativeFloat
    spin_vector: tuple[float, float, float]
    speed: NonNegativeFloat

    @property
    def horizontal_spin(self) -> float:
        """Return magnitude of spin component in the orbital plane.

        Returns:
            float: Horizontal spin magnitude.
        """

        return np.sqrt(self.spin_vector[0] ** 2 + self.spin_vector[1] ** 2)

    @property
    def vertical_spin(self) -> float:
        """Return spin component along the z-axis.

        Returns:
            float: Vertical spin component.
        """

        return self.spin_vector[2]


BlackHoles: TypeAlias = list[BlackHole]


class BlackHoleGenerator(BaseModel, frozen=True):
    """Parametric generator for black-hole samples.

    Draws masses and spin angles/magnitudes from configured distributions and
    converts sampled spherical spin parameters into Cartesian spin components.
    """

    mass_distribution: Distribution = Uniform(low=BH_MASS_LB, high=PISN_LB)
    spin_magnitude_distribution: Distribution = Uniform(low=0, high=BH_SPIN_UB)
    phi_distribution: Distribution = Uniform(low=0, high=2 * np.pi)
    theta_distribution: Distribution = Uniform(low=0, high=np.pi)

    @field_validator("spin_magnitude_distribution", mode="before")
    @classmethod
    def validate_spin_magnitude_distribution(cls, v):
        """Validate spin-magnitude support is within [0, 1].

        Args:
            v: Candidate distribution object.

        Returns:
            Any: Unchanged validated distribution.

        Raises:
            ValueError: If distribution support falls outside [0, 1].
        """

        if (v.min < 0) or (v.max > 1):
            raise ValueError("Spin magnitude distribution must be within the range [0, 1].")

        return v

    @field_validator("phi_distribution", mode="before")
    @classmethod
    def validate_phi_distribution(cls, v):
        """Validate azimuthal-angle support is within [0, 2*pi].

        Args:
            v: Candidate distribution object.

        Returns:
            Any: Unchanged validated distribution.

        Raises:
            ValueError: If distribution support falls outside [0, 2*pi].
        """

        if (v.min < 0) or (v.max > 2 * np.pi):
            raise ValueError("Phi distribution must be within the range [0, 2 * pi].")

        return v

    @field_validator("theta_distribution", mode="before")
    @classmethod
    def validate_theta_distribution(cls, v):
        """Validate polar-angle support is within [0, pi].

        Args:
            v: Candidate distribution object.

        Returns:
            Any: Unchanged validated distribution.

        Raises:
            ValueError: If distribution support falls outside [0, pi].
        """

        if (v.min < 0) or (v.max > np.pi):
            raise ValueError("Theta distribution must be within the range [0, pi].")

        return v

    def draw(self, size: int = 1, random_state: Optional[int] = None) -> BlackHoles:
        """Draw black-hole samples from configured parameter distributions.

        Args:
            size (int): Number of black holes to generate.
            random_state (Optional[int]): Random seed for reproducibility.

        Returns:
            BlackHoles: Generated black-hole list.
        """

        seed_sequence = np.random.SeedSequence(random_state)
        random_states = seed_sequence.generate_state(4)

        masses = self.mass_distribution.draw(size=size, random_state=random_states[0])
        spin_magnitudes = self.spin_magnitude_distribution.draw(size=size, random_state=random_states[1])
        phis = self.phi_distribution.draw(size=size, random_state=random_states[2])
        thetas = self.theta_distribution.draw(size=size, random_state=random_states[3])

        return self.build_black_holes(
            masses=masses,
            spin_magnitudes=spin_magnitudes,
            phis=phis,
            thetas=thetas,
        )

    @staticmethod
    def build_black_holes(
        masses: list[float],
        spin_magnitudes: list[float],
        phis: list[float],
        thetas: list[float],
    ) -> BlackHoles:
        """Assemble black-hole objects from sampled scalar parameters.

        Args:
            masses (list[float]): Mass samples.
            spin_magnitudes (list[float]): Spin magnitude samples.
            phis (list[float]): Azimuthal angles.
            thetas (list[float]): Polar angles.

        Returns:
            BlackHoles: Constructed black-hole objects with Cartesian spin vectors.
        """

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
    """Finite black-hole population with replacement-based resampling.

    Holds a fixed collection of black holes and supports random draws with
    replacement for Monte Carlo workflows. Can also be constructed from simulation
    remnant tables.
    """

    black_holes: BlackHoles

    def draw(self, size: int = 1, random_state: Optional[int] = None) -> BlackHoles:
        """Sample black holes with replacement from a stored population.

        Args:
            size (int): Number of black holes to draw.
            random_state (Optional[int]): Random seed for reproducibility.

        Returns:
            BlackHoles: Drawn black-hole list.
        """

        rng = np.random.default_rng(random_state)
        return rng.choice(self.black_holes, size=size, replace=True).tolist()

    @classmethod
    def from_simulation_results(
        cls,
        df: pd.DataFrame,
        phi_distribution: Distribution = Uniform(low=0, high=2 * np.pi),
        theta_distribution: Distribution = Uniform(low=0, high=np.pi),
        random_state: Optional[int] = None,
    ) -> "BlackHolePopulation":
        """Build a black-hole population from remnant columns in a simulation dataframe.

        Args:
            df (pd.DataFrame): Simulation dataframe containing remnant mass, spin, and kick.
            phi_distribution (Distribution): Distribution for azimuthal angles.
            theta_distribution (Distribution): Distribution for polar angles.
            random_state (Optional[int]): Random seed for reproducibility.

        Returns:
            BlackHolePopulation: Population object constructed from remnant entries.
        """

        seed_sequence = np.random.SeedSequence(random_state)
        random_states = seed_sequence.generate_state(2)

        phis = phi_distribution.draw(size=len(df), random_state=random_states[0])
        thetas = theta_distribution.draw(size=len(df), random_state=random_states[1])

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
