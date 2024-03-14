from typing import Callable, Optional

import numpy as np

import core.math
import core.utils
import logger
import schemas.binary
import schemas.common


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
        self._only_up_aligned_spin = settings.only_up_aligned_spin
        self._is_mass_injected = is_mass_injected
        self._mass_domain = settings.mass
        self._mass_ratio_domain = settings.mass_ratio

        if mass_from_pdf and mass_ratio_from_pdf:
            raise ValueError("Both mass_from_pdf and mass_ratio_from_pdf exist.")

        if mass_from_pdf:
            local_logger.info("Using mass_from_pdf.")
            self._mass_generator = mass_from_pdf
        else:
            self._mass_generator = core.math.get_generator_from_domain(settings.mass)

        if mass_ratio_from_pdf:
            local_logger.info("Using mass_ratio_from_pdf.")
            self._mass_ratio_generator = mass_ratio_from_pdf
        else:
            self._mass_ratio_generator = core.math.get_generator_from_domain(settings.mass_ratio)

        self._spin_generator = core.math.get_generator_from_domain(settings.spin)
        self._phi_generator = core.math.get_generator_from_domain(settings.phi)
        self._theta_generator = self._get_theta_generator(settings.theta)

        local_logger.info(
            "Constructed a binary generator: mass injected: %s, settings: %s",
            self._is_mass_injected,
            settings,
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

        m1, m2 = self._get_masses()
        mass_ratio = m1 / m2

        # mass_ratio = self._mass_ratio_generator()
        # m1, m2 = self._get_masses_from_mass_ratio(mass_ratio)

        return schemas.binary.Binary(mass_ratio, chi1, chi2, m1, m2)

    @staticmethod
    def _get_theta_generator(theta_domain: schemas.common.Domain) -> Callable:
        """
        Get theta generator.

        Args:
        -----
            theta_domain (schemas.common.Domain):
                The domain of theta.

        Returns:
        -----
            generate_theta (Callable):
                A function that generates theta.
        """

        def generate_theta() -> float:
            """
            Generate theta.

            Returns:
            -----
                theta (float):
                    The generated theta.
            """

            return np.arccos(-1 + 2 * np.random.uniform(theta_domain.low / np.pi, theta_domain.high / np.pi))

        return generate_theta

    def _get_spin(self) -> tuple[float, float, float]:
        """
        Get spin.

        Returns:
        -----
            spin (np.ndarray):
                The generated spin.
        """

        spin = self._spin_generator()
        if self._is_spin_aligned:
            if self._only_up_aligned_spin:
                return (0.0, 0.0, spin)
            else:
                direction = np.random.choice([-1, 1])
                return (0.0, 0.0, direction * spin)
        else:
            phi = self._phi_generator()
            theta = self._theta_generator()
            univ = core.math.sph2cart(theta, phi)
            return tuple(spin * univ)

    def _get_masses(self) -> tuple[Optional[float], Optional[float]]:
        """
        Get masses.

        Returns:
        -----
            m_1 (Optional[float]):
                Mass of the heavier black hole.

            m_2 (Optional[float]):
                Mass of the lighter black hole.
        """

        if not self._is_mass_injected:
            return (None, None)

        masses = (self._mass_generator(), self._mass_generator())
        m_1, m_2 = max(masses), min(masses)

        mass_ratio = m_1 / m_2
        if not core.math.is_in_bounds(mass_ratio, self._mass_ratio_domain):
            return self._get_masses()

        return (m_1, m_2)

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
