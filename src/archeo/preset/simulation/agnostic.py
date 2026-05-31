from archeo.constants.enum import Fits
from archeo.data_structures.distribution import Uniform
from archeo.data_structures.physics.binary import BinaryGenerator
from archeo.data_structures.physics.black_hole import BlackHoleGenerator
from archeo.data_structures.physics.simulation import PipelineOutput
from archeo.postprocessing.dataframe import convert_simulated_binaries_to_dataframe
from archeo.simulation.simulate_merger import simulate_black_hole_mergers
from archeo.utils.parallel import get_n_workers


def simulate_agnostic_precession_spin_binaries(
    size: int = 1000, n_workers: int = 1, random_state: int = 42
) -> PipelineOutput:
    """Simulate agnostic binaries with precession spins.

    Args:
        size (int): Number of binaries to simulate.
        n_workers (int): Number of workers. Use `-1` for all available cores.
        random_state (int): Seed for reproducibility.

    Returns:
        PipelineOutput: Tuple containing:
            - pd.DataFrame: Simulated binary/remnant records.
            - BinaryGenerator: Generator used for simulation.
    """

    n_workers = get_n_workers(n_workers)

    bh_generator = BlackHoleGenerator(mass_distribution=Uniform(low=5, high=200))
    binary_generator = BinaryGenerator(
        primary_black_hole_source=bh_generator,
        secondary_black_hole_source=bh_generator,
        is_aligned_spin=False,
    )
    black_hole_mergers = simulate_black_hole_mergers(
        binary_generator, Fits.NRSUR7DQ4REMNANT, size, n_workers, random_state
    )
    df_binaries = convert_simulated_binaries_to_dataframe(black_hole_mergers)
    return df_binaries, binary_generator


def simulate_agnostic_aligned_spin_binaries(
    size: int = 1000, n_workers: int = 1, random_state: int = 42
) -> PipelineOutput:
    """Simulate agnostic binaries with aligned spins.

    Args:
        size (int): Number of binaries to simulate.
        n_workers (int): Number of workers. Use `-1` for all available cores.
        random_state (int): Seed for reproducibility.

    Returns:
        PipelineOutput: Tuple containing:
            - pd.DataFrame: Simulated binary/remnant records.
            - BinaryGenerator: Generator used for simulation.
    """

    n_workers = get_n_workers(n_workers)

    bh_generator = BlackHoleGenerator(mass_distribution=Uniform(low=5, high=200))
    binary_generator = BinaryGenerator(
        primary_black_hole_source=bh_generator,
        secondary_black_hole_source=bh_generator,
        is_aligned_spin=True,
    )
    black_hole_mergers = simulate_black_hole_mergers(
        binary_generator, Fits.NRSUR3DQ8REMNANT, size, n_workers, random_state
    )
    df_binaries = convert_simulated_binaries_to_dataframe(black_hole_mergers)
    return df_binaries, binary_generator
