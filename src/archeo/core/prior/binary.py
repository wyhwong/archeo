from typing import Callable, Optional

import numpy as np

import archeo.core.math
import archeo.core.utils
import archeo.core.prior.mahapatra
import archeo.logger
import archeo.schemas.binary
import archeo.schemas.common


local_logger = archeo.logger.get_logger(__name__)


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
        settings: archeo.schemas.binary.BinarySettings,
        is_mass_injected: bool,
        is_mahapatra: bool,
    ) -> None:
        """
        Initialize the binary generator.

        Args:
        -----
            settings (archeo.schemas.binary.BinarySettings):
                Binary settings.

            is_mass_injected (bool):
                Whether to inject mass.

            is_mahapatra (bool):
                Whether to use Mahapatra's mass function.

        Returns:
        -----
            None
        """

        self._is_spin_aligned = settings.is_spin_aligned
        self._only_up_aligned_spin = settings.only_up_aligned_spin
        self._is_mass_injected = is_mass_injected
        self._is_mahapatra = is_mahapatra
        self._mass_domain = settings.mass
        self._mass_ratio_domain = settings.mass_ratio

        if self._is_mahapatra:
            self._mass_generator = archeo.core.prior.mahapatra.get_mass_func_from_mahapatra(settings.mass)
        else:
            self._mass_generator = archeo.core.math.get_generator_from_domain(settings.mass)

        self._spin_generator = archeo.core.math.get_generator_from_domain(settings.spin)
        self._phi_generator = archeo.core.math.get_generator_from_domain(settings.phi)
        self._theta_generator = self._get_theta_generator(settings.theta)

        local_logger.info(
            "Constructed a binary generator: mass injected: %s, settings: %s",
            self._is_mass_injected,
            settings,
        )

    def __call__(self) -> archeo.schemas.binary.Binary:
        """
        Generate a binary.

        Returns:
        -----
            binary (archeo.schemas.binary.Binary):
                The generated binary.
        """

        chi1, chi2 = self._get_spin(), self._get_spin()

        m1, m2 = self._get_masses()
        mass_ratio = m1 / m2

        return archeo.schemas.binary.Binary(mass_ratio, chi1, chi2, m1, m2)

    @staticmethod
    def _get_theta_generator(theta_domain: archeo.schemas.common.Domain) -> Callable:
        """
        Get theta generator.

        Args:
        -----
            theta_domain (archeo.schemas.common.Domain):
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
            univ = archeo.core.math.sph2cart(theta, phi)
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

        # Case: mass is not injected
        if not self._is_mass_injected:
            return (None, None)

        # Case: mass is injected
        masses = (self._mass_generator(), self._mass_generator())
        m_1, m_2 = max(masses), min(masses)

        # Check whether the mass ratio is in the domain
        # If not, resample the masses (recursion)
        mass_ratio = m_1 / m_2
        if not archeo.core.math.is_in_bounds(mass_ratio, self._mass_ratio_domain):
            return self._get_masses()

        return (m_1, m_2)
