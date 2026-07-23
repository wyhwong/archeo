from abc import abstractmethod
from typing import Optional, TypeAlias

import numpy as np
from pydantic import BaseModel, NonNegativeFloat, PositiveFloat

from archeo.constants.physics import BH_MASS_LB, BH_SPIN_UB, PISN_LB
from archeo.data_structures.distribution import Uniform
from archeo.data_structures.math import Domain
from archeo.data_structures.physics.black_hole import BlackHole, BlackHoleGenerator, BlackHoleSource
from archeo.data_structures.type_alias import Distribution
from archeo.utils.logger import get_logger


LOGGER = get_logger(__name__)


class Binary(BaseModel, frozen=True):
    """Container for a compact-binary system with component masses and spins.

    Provides derived properties commonly used in GW inference, including mass
    ratio, effective aligned spin, and precession spin proxy.
    """

    primary_black_hole: BlackHole
    secondary_black_hole: BlackHole

    @property
    def mass_ratio(self) -> PositiveFloat:
        """Return binary mass ratio as primary mass divided by secondary mass.

        Returns:
            PositiveFloat: Mass ratio value.
        """

        return self.primary_black_hole.mass / self.secondary_black_hole.mass

    @property
    def precession_spin(self) -> NonNegativeFloat:
        """Compute effective precession spin proxy for the binary.

        Returns:
            NonNegativeFloat: Precession spin statistic derived from horizontal spins
            and mass ratio.
        """

        q = self.primary_black_hole.mass / self.secondary_black_hole.mass
        a1h = self.primary_black_hole.horizontal_spin
        a2h = self.secondary_black_hole.horizontal_spin
        return np.maximum(a1h, (4 / q + 3) / (3 / q + 4) / q * a2h)

    @property
    def effective_spin(self) -> float:
        """Compute mass-weighted effective aligned spin for the binary.

        Returns:
            NonNegativeFloat: Effective spin value.
        """

        m1 = self.primary_black_hole.mass
        m2 = self.secondary_black_hole.mass
        a1z = self.primary_black_hole.vertical_spin
        a2z = self.secondary_black_hole.vertical_spin
        return (a1z * m1 + a2z * m2) / (m1 + m2)


Binaries: TypeAlias = list[Binary]


class BinaryGeneratorBase(BaseModel, frozen=True):
    """Abstract interface for binary-population samplers.

    Subclasses implement `draw` to generate physically valid binary systems
    according to a specific population model.
    """

    is_aligned_spin: bool = False

    @abstractmethod
    def draw(self, size: int = 1) -> Binaries:
        """Generate binary samples.

        Args:
            size (int): Number of binaries to generate.

        Returns:
            Binaries: List of generated binary objects.
        """

    @staticmethod
    def _apply_aligned_spin_to_binaries(binaries: Binaries, random_state: Optional[int] = None) -> Binaries:
        """Project spins onto the z-axis with random sign for aligned-spin mode.

        Args:
            binaries (Binaries): Input binaries with generic spin vectors.
            random_state (Optional[int]): Random seed for reproducibility.

        Returns:
            Binaries: Binaries whose spin vectors are aligned or anti-aligned with z-axis.
        """

        size = len(binaries)
        rng = np.random.default_rng(random_state)
        direction_bh1 = rng.choice([-1, 1], size=size, replace=True)
        direction_bh2 = rng.choice([-1, 1], size=size, replace=True)
        binaries = [
            Binary(
                primary_black_hole=BlackHole(
                    mass=b.primary_black_hole.mass,
                    spin_magnitude=b.primary_black_hole.spin_magnitude,
                    spin_vector=(0.0, 0.0, b.primary_black_hole.spin_magnitude * direction_bh1[i]),
                    speed=b.primary_black_hole.speed,
                ),
                secondary_black_hole=BlackHole(
                    mass=b.secondary_black_hole.mass,
                    spin_magnitude=b.secondary_black_hole.spin_magnitude,
                    spin_vector=(0.0, 0.0, b.secondary_black_hole.spin_magnitude * direction_bh2[i]),
                    speed=b.secondary_black_hole.speed,
                ),
            )
            for i, b in enumerate(binaries)
        ]
        LOGGER.info("Applied aligned spin configuration to the generated binaries.")
        return binaries


class BinaryGenerator(BinaryGeneratorBase):
    """Binary sampler built from component black-hole populations.

    Draws primary/secondary components from configured sources, enforces mass
    ordering and optional mass-ratio constraints, and optionally applies aligned-spin
    projection.
    """

    primary_black_hole_source: BlackHoleSource
    secondary_black_hole_source: BlackHoleSource
    mass_ratio_domain: Domain = Domain(low=1.0, high=6.0)
    enforce_source_binding: bool = False

    def draw(self, size: int = 1, random_state: Optional[int] = None) -> Binaries:
        """Generate binaries from configured sources under mass-ratio constraints.

        Args:
            size (int): Number of binaries to generate.
            random_state (Optional[int]): Random seed for reproducibility.

        Returns:
            Binaries: Generated binary list satisfying ordering and ratio criteria.
        """

        binaries = []
        n_step = 0
        seed_sequence = np.random.SeedSequence(random_state)

        while len(binaries) < size:
            n_step += 1
            remaining_size = size - len(binaries)
            random_states = seed_sequence.generate_state(2)

            primary_black_holes = self.primary_black_hole_source.draw(
                size=remaining_size, random_state=random_states[0]
            )
            secondary_black_holes = self.secondary_black_hole_source.draw(
                size=remaining_size, random_state=random_states[1]
            )

            for p_bh, s_bh in zip(primary_black_holes, secondary_black_holes):
                if (not self.enforce_source_binding) and (p_bh.mass < s_bh.mass):
                    p_bh, s_bh = s_bh, p_bh

                if p_bh.mass < s_bh.mass:
                    continue

                if not self.mass_ratio_domain.contains(p_bh.mass / s_bh.mass):
                    continue

                binaries.append(Binary(primary_black_hole=p_bh, secondary_black_hole=s_bh))

            LOGGER.info("Step %d: Generated %d binaries so far.", n_step, len(binaries))

        LOGGER.info("Finished generating %d binaries after %d steps.", size, n_step)

        if self.is_aligned_spin:
            binaries = self._apply_aligned_spin_to_binaries(binaries, random_state=seed_sequence.generate_state(1)[0])

        return binaries


class MassRatioBasedBinaryGenerator(BinaryGeneratorBase):
    """Binary sampler using primary-mass and mass-ratio distributions.

    Generates binaries by sampling primary masses and then deriving secondary
    masses through a configured mass-ratio model within domain bounds.
    """

    mass_ratio_distribution: Distribution = Uniform(low=1.0, high=6.0)
    primary_mass_distribution: Distribution = Uniform(low=BH_MASS_LB, high=PISN_LB)
    secondary_mass_domain: Domain = Domain(low=BH_MASS_LB, high=PISN_LB)
    spin_magnitude_distribution: Distribution = Uniform(low=0.0, high=BH_SPIN_UB)
    phi_distribution: Distribution = Uniform(low=0, high=2 * np.pi)
    theta_distribution: Distribution = Uniform(low=0, high=np.pi)

    def draw(self, size: int = 1, random_state: Optional[int] = None) -> Binaries:
        """Generate binaries by sampling primary masses and mass-ratio distribution.

        Args:
            size (int): Number of binaries to generate.
            random_state (Optional[int]): Random seed for reproducibility.

        Returns:
            Binaries: Generated binary list.
        """

        seed_sequence = np.random.SeedSequence(random_state)
        random_states = seed_sequence.generate_state(8)

        primary_masses, secondary_masses = self.sample_binary_masses(size, random_state=random_states[0])
        primary_bhs = BlackHoleGenerator.build_black_holes(
            masses=primary_masses,
            spin_magnitudes=self.spin_magnitude_distribution.draw(size, random_state=random_states[1]),
            phis=self.phi_distribution.draw(size, random_state=random_states[2]),
            thetas=self.theta_distribution.draw(size, random_state=random_states[3]),
        )
        secondary_bhs = BlackHoleGenerator.build_black_holes(
            masses=secondary_masses,
            spin_magnitudes=self.spin_magnitude_distribution.draw(size, random_state=random_states[4]),
            phis=self.phi_distribution.draw(size, random_state=random_states[5]),
            thetas=self.theta_distribution.draw(size, random_state=random_states[6]),
        )
        binaries = [
            Binary(primary_black_hole=bh1, secondary_black_hole=bh2) for bh1, bh2 in zip(primary_bhs, secondary_bhs)
        ]

        if self.is_aligned_spin:
            binaries = self._apply_aligned_spin_to_binaries(
                binaries, random_state=random_states[7] if random_state is not None else None
            )

        return binaries

    def sample_binary_masses(self, size: int, random_state: Optional[int] = None) -> tuple[list[float], list[float]]:
        """Sample primary and secondary masses consistent with mass-domain limits.

        Args:
            size (int): Number of binaries.
            random_state (Optional[int]): Random seed for reproducibility.

        Returns:
            tuple[list[float], list[float]]: Primary mass list and secondary mass list.
        """

        seed_sequence = np.random.SeedSequence(random_state)
        random_states = seed_sequence.generate_state(2)

        mass_ratios = np.asarray(self.mass_ratio_distribution.draw(size=size, random_state=random_states[0]))
        primary_masses = np.asarray(self.primary_mass_distribution.draw(size=size, random_state=random_states[1]))
        secondary_masses = primary_masses / mass_ratios
        n_step = 1

        while (mask := self.secondary_mass_domain.not_contains(secondary_masses)).any():
            LOGGER.info("Step %d: Generated %d binaries so far.", n_step, len(primary_masses) - mask.sum())
            n_step += 1
            primary_masses[mask] = self.primary_mass_distribution.draw(
                size=mask.sum(), random_state=seed_sequence.generate_state(1)[0]
            )
            secondary_masses[mask] = primary_masses[mask] / mass_ratios[mask]

        LOGGER.info("Finished generating %d binaries after %d steps.", size, n_step)
        return primary_masses, secondary_masses
