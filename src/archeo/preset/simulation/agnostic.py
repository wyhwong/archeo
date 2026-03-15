from typing import Optional

from archeo.constants.enum import Fits
from archeo.data_structures.distribution import Uniform
from archeo.data_structures.physics.binary import BinaryGenerator
from archeo.data_structures.physics.black_hole import BlackHoleGenerator
from archeo.postprocessing.dataframe import convert_simulated_binaries_to_dataframe
from archeo.simulation.simulate_merger import simulate_black_hole_mergers


def simulate_agnostic_precession_spin_binaries(size: int = 2000000, n_workers: Optional[int] = None):
    """Simulate a population of agnostic precession spin binaries.

    Args:
        size (int): The number of binaries to simulate. Default is 2,000,000.
        n_workers (Optional[int]): The number of workers to use for parallel processing.
            If None, it will use all available cores.

    Returns:
        pd.DataFrame: A DataFrame containing the simulated binaries and their properties.
    """

    bh_generator = BlackHoleGenerator(mass=Uniform(5, 200))
    binary_generator = BinaryGenerator(
        primary_black_hole_source=bh_generator,
        secondary_black_hole_source=bh_generator,
        is_aligned_spin=False,
    )
    binaries, remnants = simulate_black_hole_mergers(binary_generator, Fits.NRSUR7DQ4REMNANT, size, n_workers)
    df_binaries = convert_simulated_binaries_to_dataframe(binaries, remnants)
    return df_binaries


def simulate_agnostic_aligned_spin_binaries(size: int = 2000000, n_workers: Optional[int] = None):
    """Simulate a population of agnostic aligned spin binaries.

    Args:
        size (int): The number of binaries to simulate. Default is 2,000,000.
        n_workers (Optional[int]): The number of workers to use for parallel processing.
            If None, it will use all available cores.

    Returns:
        pd.DataFrame: A DataFrame containing the simulated binaries and their properties.
    """

    bh_generator = BlackHoleGenerator(mass=Uniform(5, 200))
    binary_generator = BinaryGenerator(
        primary_black_hole_source=bh_generator,
        secondary_black_hole_source=bh_generator,
        is_aligned_spin=True,
    )
    binaries, remnants = simulate_black_hole_mergers(binary_generator, Fits.NRSUR3DQ8REMNANT, size, n_workers)
    df_binaries = convert_simulated_binaries_to_dataframe(binaries, remnants)
    return df_binaries
