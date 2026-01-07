import logging
from dataclasses import asdict
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from archeo.constants import Columns as C
from archeo.constants import Suffixes as S
from archeo.core.mahapatra import get_mahapatra_mass_fn
from archeo.schema import Binary, Domain, Event, Interface, PriorConfig
from archeo.utils.helper import pre_release
from archeo.utils.math import sph2cart
from archeo.utils.parallel import multithread_run


logger = logging.getLogger(__name__)


def _extended_draw(domains: list[Domain]) -> float:

    indices = [i for i in range(len(domains))]
    weights = np.array([domain.width for domain in domains])
    p = weights / sum(weights)

    def draw() -> float:
        """Draws a random value from the extended domain"""

        idx = np.random.choice(indices, p=p)
        return domains[idx].draw()

    return draw


class Simulator(Interface):
    """Simulator class to simulate binary black hole mergers"""

    def __init__(self, prior_config: PriorConfig):
        """Initializes the simulator with the given prior configuration

        Args:
            prior_config (PriorConfig): The prior configuration to use
        """

        self._prior_config = prior_config

        self._fits = self._prior_config.fits.load()
        self._n_samples = self._prior_config.n_samples

        if self._prior_config.is_mahapatra_mass_func:
            self._m1_fn = get_mahapatra_mass_fn(mass=self._prior_config.m_1)
            self._m2_fn = get_mahapatra_mass_fn(mass=self._prior_config.m_2)
        else:
            self._m1_fn = self._prior_config.m_1.draw
            self._m2_fn = self._prior_config.m_2.draw

        # Dummy functions
        self._is_remnant_1 = False
        self._r1_fn = lambda: (0, 0, 0)
        self._is_remnant_2 = False
        self._r2_fn = lambda: (0, 0, 0)

        self._chi1_fns = {
            "magnitude": self._prior_config.a_1.draw,
            "theta": self._prior_config.theta_1.draw,
            "phi": self._prior_config.phi_1.draw,
        }
        self._chi2_fns = {
            "magnitude": self._prior_config.a_2.draw,
            "theta": self._prior_config.theta_2.draw,
            "phi": self._prior_config.phi_2.draw,
        }

        self._m2_bounds = self._prior_config.m_2
        self._q_bounds = self._prior_config.mass_ratio

        self._is_spin_aligned = self._prior_config.is_spin_aligned
        self._only_up_aligned_spin = self._prior_config.is_only_up_aligned_spin
        self._is_uniform_in_q = self._prior_config.is_uniform_in_mass_ratio

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
            k_1=b.k_1,
            k_2=b.k_2,
            m_ret=m_ret,
            m_ret_err=m_ret_err,
            v_f=v_f,
            v_f_err=v_f_err,
            chi_1=b.chi_1,
            chi_2=b.chi_2,
            chi_f=chi_f,
            chi_f_err=chi_f_err,
        )

    def replace_draw_function(
        self,
        param: Literal["m_1", "m_2", "a_1", "a_2"],
        domains: list[Domain],
    ) -> None:
        """Replaces the draw function of the given parameter with a new one

        Args:
            param (str): The parameter to replace the draw function of
            domains (list[Domain]): The list of domains to draw from
        """

        draw_fn = _extended_draw(domains)

        if param == "m_1":
            self._m1_fn = draw_fn
        elif param == "m_2":
            self._m2_fn = draw_fn
        elif param == "a_1":
            self._chi1_fns["magnitude"] = draw_fn
        elif param == "a_2":
            self._chi2_fns["magnitude"] = draw_fn
        else:
            raise ValueError(f"Unknown parameter: {param}")

        logger.warning(
            "Using piecewise domains for parameter %s. " "This will override the prior configuration.",
            param,
        )

    def simulate(self, use_threads=True, n_threads: Optional[int] = None) -> pd.DataFrame:
        """Simulates multiple binary black hole merger events"""

        if use_threads:
            events = multithread_run(self, [{} for _ in range(self._n_samples)], n_threads)
        else:
            events = [self() for _ in tqdm(range(self._n_samples))]

        df = pd.DataFrame([asdict(event) for event in events])
        return df

    def _get_binary(self) -> Binary:
        """Draws a binary from the prior distribution

        Returns:
            Binary: The drawn binary
        """

        if self._is_uniform_in_q:
            m_1, a_1, k_1, m_2, a_2, k_2 = self._get_params_unif_q()
        else:
            m_1, a_1, k_1, m_2, a_2, k_2 = self._get_params_non_unif_q()

        chi_uv_1, chi_uv_2 = self._get_spin_uv(self._chi1_fns), self._get_spin_uv(self._chi2_fns)
        chi_1 = (chi_uv_1[0] * a_1, chi_uv_1[1] * a_1, chi_uv_1[2] * a_1)
        chi_2 = (chi_uv_2[0] * a_2, chi_uv_2[1] * a_2, chi_uv_2[2] * a_2)

        return Binary(m_1=m_1, m_2=m_2, chi_1=chi_1, chi_2=chi_2, k_1=k_1, k_2=k_2)

    def _get_spin_uv(self, fns: dict[str, Callable]) -> tuple[float, float, float]:
        """Draws the spin of the binary (unit vector)

        Args:
            fns (dict[str, Callable]): The functions to draw the spin

        Returns:
            tuple[float, float, float]: The drawn spin unit vector
        """

        if self._is_spin_aligned:
            if self._only_up_aligned_spin:
                return (0, 0, 1)

            direction = np.random.choice([-1, 1])
            return (0, 0, direction)

        theta = np.arccos(-1 + 2 * fns["theta"]())
        phi = fns["phi"]() * np.pi
        univ = sph2cart(theta=theta, phi=phi)
        return tuple(univ)

    def _get_params_non_unif_q(self) -> tuple[float, float, float, float, float, float]:
        """Draws the masses of the binary from the mass functions

        Returns:
            tuple[float, float]: The drawn masses
        """

        if not self._is_remnant_1 and not self._is_remnant_2:
            m_1, m_2 = (self._m1_fn(), self._m2_fn())
            a_1, a_2 = self._chi1_fns["magnitude"](), self._chi2_fns["magnitude"]()
            k_1, k_2 = (0, 0)
        elif self._is_remnant_1 and self._is_remnant_2:
            m_1, a_1, k_1 = self._r1_fn()
            m_2, a_2, k_2 = self._r2_fn()
        elif self._is_remnant_1:
            m_1, a_1, k_1 = self._r1_fn()
            m_2, a_2, k_2 = self._m2_fn(), self._chi2_fns["magnitude"](), 0.0
        # Only self._is_remnant_2
        else:
            m_1, a_1, k_1 = self._m1_fn(), self._chi1_fns["magnitude"](), 0.0
            m_2, a_2, k_2 = self._r2_fn()

        # If not, resample the masses (recursion)
        if self._prior_config.is_masses_swappable:
            m_1, m_2 = max(m_1, m_2), min(m_1, m_2)

        q = m_1 / m_2
        # Check whether the mass ratio is in the domain
        if not self._q_bounds.contain(q):
            return self._get_params_non_unif_q()

        return (m_1, a_1, k_1, m_2, a_2, k_2)

    def _get_params_unif_q(self) -> tuple[float, float, float, float, float, float]:
        """Draws the masses of the binary from the mass ratio function

        Returns:
            tuple[float, float]: The drawn masses
        """

        q = self._q_bounds.draw()
        m_1 = self._m1_fn()
        m_2 = m_1 / q

        # Check whether the masses are in the domain
        # If not, resample the masses (recursion)
        if not self._m2_bounds.contain(m_2):
            return self._get_params_unif_q()

        a_1 = self._chi1_fns["magnitude"]()
        a_2 = self._chi2_fns["magnitude"]()
        k_1, k_2 = (0.0, 0.0)

        return (m_1, a_1, k_1, m_2, a_2, k_2)

    @staticmethod
    @pre_release
    def get_draw_func(df: pd.DataFrame) -> Callable[[], tuple[float, float, float]]:
        """Returns a function to draw remnant properties from the dataframe"""

        def draw() -> tuple[float, float, float]:
            """Draws the remnant mass and spin"""

            idx = np.random.random_integers(low=0, high=len(df) - 1)
            return (
                df.loc[idx, S.FINAL(C.MASS)],
                df.loc[idx, S.FINAL(C.SPIN_MAG)],
                df.loc[idx, S.FINAL(C.KICK)],
            )

        return draw

    @pre_release
    def use_remnant_results(
        self,
        df_bh1: Optional[pd.DataFrame] = None,
        df_bh2: Optional[pd.DataFrame] = None,
        kick_limit: Optional[float] = None,
    ) -> None:
        """Uses the remnant results from the given file

        Args:
            df_bh1 (Optional[pd.DataFrame]): The dataframe for black hole 1 remnant results
            df_bh2 (Optional[pd.DataFrame]): The dataframe for black hole 2 remnant results
            kick_limit (Optional[float]): The maximum kick velocity allowed
        """

        if self._is_uniform_in_q:
            raise ValueError("Cannot use remnant results with uniform mass ratio prior")

        if df_bh1 is not None:
            df_bh1 = df_bh1[df_bh1[S.FINAL(C.KICK)] <= (kick_limit or np.inf)][
                [S.FINAL(C.SPIN_MAG), S.FINAL(C.MASS), S.FINAL(C.KICK)]
            ].reset_index(drop=True)
            self._is_remnant_1 = True
            self._r1_fn = self.get_draw_func(df_bh1)
            logger.warning(
                "Using remnant results for black hole 1. "
                "This will override the prior configuration for black hole 1."
            )

        if df_bh2 is not None:
            df_bh2 = df_bh2[df_bh2[S.FINAL(C.KICK)] <= (kick_limit or np.inf)][
                [S.FINAL(C.SPIN_MAG), S.FINAL(C.MASS), S.FINAL(C.KICK)]
            ].reset_index(drop=True)
            self._is_remnant_2 = True
            self._r2_fn = self.get_draw_func(df_bh2)
            logger.warning(
                "Using remnant results for black hole 2. "
                "This will override the prior configuration for black hole 2."
            )

    @property
    def model_name(self) -> str:
        """Returns the name of the model used for simulation"""

        return self._prior_config.fits.value
