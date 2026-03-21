import pyinstrument

from archeo.constants.enum import Fits
from archeo.preset.simulation.second_generation import (
    simulate_second_generation_aligned_spin_binaries,
    simulate_second_generation_precession_spin_binaries,
)


@pyinstrument.profile()
def profile_second_generation_aligned_spin_simulation():
    """Profile the second generation aligned spin binary simulation.
    This will clean up the surfinbh data before running the simulation."""

    Fits.clean_up_surfinbh_data()
    simulate_second_generation_aligned_spin_binaries(size=10000, n_workers=1)


@pyinstrument.profile()
def profile_second_generation_aligned_spin_simulation_parallel():
    """Profile the second generation aligned spin binary simulation in parallel."""

    simulate_second_generation_aligned_spin_binaries(size=10000, n_workers=-1)


@pyinstrument.profile()
def profile_second_generation_precession_spin_simulation():
    """Profile the second generation precession spin binary simulation.
    This will clean up the surfinbh data before running the simulation."""

    Fits.clean_up_surfinbh_data()
    simulate_second_generation_precession_spin_binaries(size=10000, n_workers=1)


@pyinstrument.profile()
def profile_second_generation_precession_spin_simulation_parallel():
    """Profile the second generation precession spin binary simulation in parallel."""

    simulate_second_generation_precession_spin_binaries(size=10000, n_workers=-1)


if __name__ == "__main__":
    profile_second_generation_aligned_spin_simulation()
    profile_second_generation_aligned_spin_simulation_parallel()
    profile_second_generation_precession_spin_simulation()
    profile_second_generation_precession_spin_simulation_parallel()
