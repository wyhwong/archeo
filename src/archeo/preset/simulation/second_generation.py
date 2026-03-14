from archeo.constants.enum import Fits
from archeo.data_structures.physics.binary import BinaryGenerator
from archeo.data_structures.physics.black_hole import BlackHoleGenerator
from archeo.data_structures.physics.simulation import PipelineOutput
from archeo.postprocessing.dataframe import convert_simulated_binaries_to_dataframe
from archeo.simulation.simulate_merger import simulate_black_hole_mergers
from archeo.utils.parallel import get_n_workers


def simulate_second_generation_precession_spin_binaries(size: int = 1000, n_workers: int = 1) -> PipelineOutput:
    """Simulate a population of second generation precession spin binaries.

    Args:
        size (int): The number of binaries to simulate.
            Default is 1,000.
        n_workers (int): The number of workers to use for parallel processing.
            Default is 1.
            If -1, it will use all available cores.

    Returns:
        SimulationOutput: tuple(pd.DataFrame, BinaryGenerator)
            The dataframe contains the simulated binaries and their properties.
            The BinaryGenerator contains the black hole generator used for the simulation.
    """

    n_workers = get_n_workers(n_workers)

    bh_generator = BlackHoleGenerator()
    binary_generator = BinaryGenerator(
        primary_black_hole_source=bh_generator,
        secondary_black_hole_source=bh_generator,
        is_aligned_spin=False,
    )
    black_hole_mergers = simulate_black_hole_mergers(binary_generator, Fits.NRSUR7DQ4REMNANT, size, n_workers)
    df_binaries = convert_simulated_binaries_to_dataframe(black_hole_mergers)
    return df_binaries, binary_generator


def simulate_second_generation_aligned_spin_binaries(size: int = 1000, n_workers: int = 1) -> PipelineOutput:
    """Simulate a population of second generation aligned spin binaries.

    Args:
        size (int): The number of binaries to simulate.
            Default is 1,000.
        n_workers (int): The number of workers to use for parallel processing.
            Default is 1.
            If -1, it will use all available cores.

    Returns:
        SimulationOutput: tuple(pd.DataFrame, BinaryGenerator)
            The dataframe contains the simulated binaries and their properties.
            The BinaryGenerator contains the black hole generator used for the simulation.
    """

    n_workers = get_n_workers(n_workers)

    bh_generator = BlackHoleGenerator()
    binary_generator = BinaryGenerator(
        primary_black_hole_source=bh_generator,
        secondary_black_hole_source=bh_generator,
        is_aligned_spin=True,
    )
    black_hole_mergers = simulate_black_hole_mergers(binary_generator, Fits.NRSUR3DQ8REMNANT, size, n_workers)
    df_binaries = convert_simulated_binaries_to_dataframe(black_hole_mergers)
    return df_binaries, binary_generator
