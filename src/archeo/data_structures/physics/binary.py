from abc import abstractmethod
from typing import TypeAlias

import numpy as np
from pydantic import BaseModel, NonNegativeFloat, PositiveFloat

from archeo.constants.physics import BH_MASS_LB, PISN_LB
from archeo.data_structures.annotation import Distribution
from archeo.data_structures.distribution import Uniform
from archeo.data_structures.math import Domain
from archeo.data_structures.physics.black_hole import BlackHole, BlackHoleGenerator, BlackHoleSource
from archeo.utils.logger import get_logger


LOGGER = get_logger(__name__)


class Binary(BaseModel, frozen=True):
    """Binary data class."""

    primary_black_hole: BlackHole
    secondary_black_hole: BlackHole

    @property
    def mass_ratio(self) -> PositiveFloat:
        """Calculate the mass ratio (q) for the binary."""

        return self.primary_black_hole.mass / self.secondary_black_hole.mass

    @property
    def precession_spin(self) -> NonNegativeFloat:
        """Calculate the precession spin parameter (chi_p) for the binary."""

        q = self.primary_black_hole.mass / self.secondary_black_hole.mass
        a1h = self.primary_black_hole.horizontal_spin
        a2h = self.secondary_black_hole.horizontal_spin
        return np.maximum(a1h, (4 / q + 3) / (3 / q + 4) / q * a2h)

    @property
    def effective_spin(self) -> NonNegativeFloat:
        """Calculate the effective spin parameter (chi_eff) for the binary."""

        m1 = self.primary_black_hole.mass
        m2 = self.secondary_black_hole.mass
        a1z = self.primary_black_hole.vertical_spin
        a2z = self.secondary_black_hole.vertical_spin
        return (a1z * m1 + a2z * m2) / (m1 + m2)


Binaries: TypeAlias = list[Binary]


class BinaryGeneratorBase(BaseModel, frozen=True):
    """Base class for binary generators."""

    is_aligned_spin: bool = False

    @abstractmethod
    def draw(self, size: int = 1) -> Binaries:
        """Generate a list of binaries based on the specified sources and mass ratio domain."""

    @staticmethod
    def _apply_aligned_spin_to_binaries(binaries: Binaries) -> Binaries:
        """Apply aligned spin configuration to the generated binaries."""

        size = len(binaries)
        direction_bh1 = np.random.choice([-1, 1], size=size)
        direction_bh2 = np.random.choice([-1, 1], size=size)
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
    """Binary generator data class."""

    primary_black_hole_source: BlackHoleSource
    secondary_black_hole_source: BlackHoleSource
    mass_ratio_domain: Domain = Domain(low=1.0, high=6.0)
    enforce_source_binding: bool = False

    def draw(self, size: int = 1) -> Binaries:
        """Generate a list of binaries based on the specified sources and mass ratio domain."""

        binaries = []
        n_step = 0

        while len(binaries) < size:
            n_step += 1

            remaining_size = size - len(binaries)
            primary_black_hole = self.primary_black_hole_source.draw(size=remaining_size)
            secondary_black_hole = self.secondary_black_hole_source.draw(size=remaining_size)

            for p_bh, s_bh in zip(primary_black_hole, secondary_black_hole):
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
            binaries = self._apply_aligned_spin_to_binaries(binaries)

        return binaries


class MassRatioBasedBinaryGenerator(BinaryGeneratorBase):
    """Another option for binary generator that sample binaries based on
    the mass ratio distribution and primary mass distribution.
    """

    mass_ratio_distribution: Distribution = Uniform(low=1.0, high=6.0)
    primary_mass_distribution: Distribution = Uniform(low=BH_MASS_LB, high=PISN_LB)
    secondary_mass_domain: Domain = Domain(low=BH_MASS_LB, high=PISN_LB)
    spin_magnitude_distribution: Distribution = Uniform(low=0.0, high=1.0)
    phi_distribution: Distribution = Uniform(low=0, high=2 * np.pi)
    theta_distribution: Distribution = Uniform(low=0, high=np.pi)

    def draw(self, size: int = 1) -> Binaries:
        """Generate a list of binaries based on the specified distributions and mass domain."""

        primary_masses, secondary_masses = self.sample_binary_masses(size)
        primary_bhs = BlackHoleGenerator.build_black_holes(
            masses=primary_masses,
            spin_magnitudes=self.spin_magnitude_distribution.draw(size),
            phis=self.phi_distribution.draw(size),
            thetas=self.theta_distribution.draw(size),
        )
        secondary_bhs = BlackHoleGenerator.build_black_holes(
            masses=secondary_masses,
            spin_magnitudes=self.spin_magnitude_distribution.draw(size),
            phis=self.phi_distribution.draw(size),
            thetas=self.theta_distribution.draw(size),
        )
        binaries = [
            Binary(primary_black_hole=bh1, secondary_black_hole=bh2) for bh1, bh2 in zip(primary_bhs, secondary_bhs)
        ]

        if self.is_aligned_spin:
            binaries = self._apply_aligned_spin_to_binaries(binaries)

        return binaries

    def sample_binary_masses(self, size: int) -> tuple[list[float], list[float]]:

        mass_ratios = self.mass_ratio_distribution.draw(size=size)
        primary_masses = self.primary_mass_distribution.draw(size=size)
        secondary_masses = primary_masses / mass_ratios
        n_step = 1

        while (mask := self.secondary_mass_domain.not_contains(secondary_masses)).any():
            LOGGER.info("Step %d: Generated %d binaries so far.", n_step, len(primary_masses) - mask.sum())
            primary_masses[mask] = self.primary_mass_distribution.draw(size=mask.sum())
            secondary_masses[mask] = primary_masses[mask] / mass_ratios[mask]
            n_step += 1

        LOGGER.info("Finished generating %d binaries after %d steps.", size, n_step)
        return primary_masses, secondary_masses
