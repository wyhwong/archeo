import numpy as np
from typing import Callable, Optional

import schemas.common
import schemas.binary
import core.utils
import core.math
import logger

local_logger = logger.get_logger(__name__)


class BinaryGenerator:
    """
    Binary generator.
    NOTE:
        Convention:
            1. Heavier black hole
            2. Lighter black hole
    """

    def __init__(
        self,
        settings: schemas.binary.BinarySettings,
        is_mass_injected: bool,
        mass_ratio_from_pdf: Optional[Callable] = None,
        mass_from_pdf: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the binary generator.

        Args:
        -----
            settings (schemas.binary.BinarySettings):
                Binary settings.

            is_mass_injected (bool):
                Whether to inject mass.

            mass_ratio_from_pdf (Optional[Callable]):
                A function that returns a mass ratio sampled from a PDF.

            mass_from_pdf (Optional[Callable]):
                A function that returns a mass sampled from a PDF.

        Returns:
        -----
            None
        """

        self._is_spin_aligned = settings.is_spin_aligned
        self._is_mass_injected = is_mass_injected
        self._mass_domain = settings.mass

        if mass_from_pdf and mass_ratio_from_pdf:
            raise ValueError("Both mass_from_pdf and mass_ratio_from_pdf exist.")

        if mass_from_pdf:
            local_logger.info("Using mass_from_pdf.")
            self._mass_generator = mass_from_pdf
        else:
            self._mass_generator = core.math.get_generator_from_domain(self.config.mass)

        if mass_ratio_from_pdf:
            local_logger.info("Using mass_ratio_from_pdf.")
            self._mass_ratio_generator = mass_ratio_from_pdf
        else:
            self._mass_ratio_generator = core.math.get_generator_from_domain(self.config.mass_ratio)

        self._spin_generator = core.math.get_generator_from_domain(self.config.spin)
        self._phi_generator = core.math.get_generator_from_domain(self.config.phi)
        self._theta_generator = core.math.get_generator_from_domain(self.config.theta)

        local_logger.info(
            "Constructed a binary generator: mass injected: %s, config: %s",
            self._is_mass_injected,
            self.config,
        )

    def __call__(self) -> schemas.binary.Binary:
        """
        Generate a binary.

        Returns:
        -----
            binary (schemas.binary.Binary):
                The generated binary.
        """

        chi1, chi2 = self._get_spin(), self._get_spin()

        mass_ratio = self._mass_ratio_generator()
        m1, m2 = self._get_masses_from_mass_ratio(mass_ratio)

        return schemas.binary.Binary(mass_ratio, chi1, chi2, m1, m2)

    def simulate(self, num: int = 1) -> list[schemas.binary.Binary]:
        """
        Simulate binaries.

        Args:
        -----
            num (int):
                Number of binaries to simulate.

        Returns:
        -----
            binaries (list[schemas.binary.Binary]):
                The simulated binaries.
        """

        return [self.__call__() for _ in range(num)]

    def _get_spin(self) -> np.ndarray:
        """
        Get spin.

        Returns:
        -----
            spin (np.ndarray):
                The generated spin.
        """

        spin = self._spin_generator()
        if self._is_spin_aligned:
            phi = 0.0
            theta = 0.0 + np.round(np.random.rand()) * np.pi
        else:
            phi = self._phi_generator()
            theta = self._theta_generator()

        univ = core.math.sph2cart(theta, phi)
        return spin * univ

    def _get_masses_from_mass_ratio(self, mass_ratio: float) -> tuple[Optional[float], Optional[float]]:
        """
        Get masses from mass ratio.

        Args:
        -----
            mass_ratio (float):
                Mass ratio of the binary.

        Returns:
        -----
            m_1 (Optional[float]):
                Mass of the heavier black hole.

            m_2 (Optional[float]):
                Mass of the lighter black hole.
        """

        if not self._is_mass_injected:
            return (None, None)

        m_1 = self._mass_generator()
        m_2 = m_1 / mass_ratio

        # Recursively generate masses until m_2 is in the domain.
        if not core.math.is_in_bounds(m_2, self._mass_domain):
            return self._get_masses_from_mass_ratio(mass_ratio)

        return (m_1, m_2)
