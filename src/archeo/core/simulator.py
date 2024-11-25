from dataclasses import asdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from archeo.core.mahapatra import get_mahapatra_mass_fn
from archeo.schema import Binary, Event, PriorConfig
from archeo.utils.executor import MultiThreadExecutor
from archeo.utils.math import sph2cart


class Simulator:
    """Simulator class to simulate binary black hole mergers"""

    def __init__(self, prior_config: PriorConfig):
        """Initializes the simulator with the given prior configuration

        Args:
            prior_config (PriorConfig): The prior configuration to use
        """

        self._prior_config = prior_config

        self._fits = self._prior_config.fits.load()
        self._n_samples = self._prior_config.n_samples

        if self._prior_config.is_mahapatra:
            self._mass_fn = get_mahapatra_mass_fn(mass=self._prior_config.mass)
        else:
            self._mass_fn = self._prior_config.mass.draw

        self._theta_fn = self._prior_config.theta.draw
        self._phi_fn = self._prior_config.phi.draw
        self._spin_fn = self._prior_config.spin.draw

        self._q_bounds = self._prior_config.mass_ratio

        self._is_spin_aligned = self._prior_config.is_spin_aligned
        self._only_up_aligned_spin = self._prior_config.is_only_up_aligned_spin

    def __call__(self) -> Event:
        """Simulates a binary black hole merger event

        Returns:
            Event: The simulated event
        """

        b = self._get_binary()

        q = b.m_1 / b.m_2
        v_f, v_f_err = self._fits.vf(q, b.chi_1, b.chi_2)
        chi_f, chi_f_err = self._fits.chif(q, b.chi_1, b.chi_2)
        m_ret, m_ret_err = self._fits.mf(q, b.chi_1, b.chi_2)

        return Event(
            m_1=b.m_1,
            m_2=b.m_2,
            m_ret=m_ret,
            m_ret_err=m_ret_err,
            v_f=v_f,
            v_f_err=v_f_err,
            chi_1=b.chi_1,
            chi_2=b.chi_2,
            chi_f=chi_f,
            chi_f_err=chi_f_err,
        )

    def simulate(self, use_threads=True) -> pd.DataFrame:
        """Simulates multiple binary black hole merger events"""

        if use_threads:
            exc = MultiThreadExecutor()
            events = exc.run(self, [{} for _ in range(self._n_samples)])
        else:
            events = [self() for _ in tqdm(range(self._n_samples))]

        df = pd.DataFrame([asdict(event) for event in events])
        return df

    def _get_binary(self) -> Binary:
        """Draws a binary from the prior distribution

        Returns:
            Binary: The drawn binary
        """

        m_1, m_2 = self._get_masses()
        chi_1, chi_2 = self._get_spin(), self._get_spin()

        return Binary(m_1=m_1, m_2=m_2, chi_1=chi_1, chi_2=chi_2)

    def _get_spin(self) -> tuple[float, float, float]:
        """Draws the spin of the binary

        Returns:
            tuple[float, float, float]: The drawn spin
        """

        spin = self._spin_fn()

        if self._is_spin_aligned:
            if self._only_up_aligned_spin:
                return (0, 0, spin)

            direction = np.random.choice([-1, 1])
            return (0, 0, direction * spin)

        theta = np.arccos(-1 + 2 * self._theta_fn())
        phi = self._phi_fn() * np.pi
        univ = sph2cart(theta=theta, phi=phi)
        return tuple(spin * univ)

    def _get_masses(self) -> tuple[float, float]:
        """Draws the masses of the binary

        Returns:
            tuple[float, float]: The drawn masses
        """

        masses = (self._mass_fn(), self._mass_fn())
        m_1, m_2 = sorted(masses, reverse=True)

        # Check whether the mass ratio is in the domain
        # If not, resample the masses (recursion)
        q = m_1 / m_2
        if not self._q_bounds.contain(q):
            return self._get_masses()

        return (m_1, m_2)
