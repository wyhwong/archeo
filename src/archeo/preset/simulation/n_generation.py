from typing import Optional

import pandas as pd

from archeo.constants.enum import Fits
from archeo.data_structures.physics.binary import BinaryGenerator
from archeo.data_structures.physics.black_hole import BlackHoleGenerator, BlackHolePopulation
from archeo.data_structures.physics.simulation import PipelineOutput
from archeo.postprocessing.dataframe import convert_simulated_binaries_to_dataframe
from archeo.simulation.simulate_merger import simulate_black_hole_mergers
from archeo.utils.parallel import get_n_workers


def simulate_multi_generation_precession_spin_binaries(
    df_bh1_binaries: pd.DataFrame,
    df_bh2_binaries: Optional[pd.DataFrame] = None,
    size: int = 1000,
    n_workers: int = 1,
    random_state: int = 42,
) -> PipelineOutput:
    """Simulate a population of multi-generation precession spin binaries.

    Args:
        df_bh1_binaries (pd.DataFrame):
            DataFrame containing the first generation black hole binaries.
        df_bh2_binaries (Optional[pd.DataFrame]):
            DataFrame containing the second generation black hole binaries.
            Default is None, meaning the second black hole will be drawn from a default distribution.
        size (int): The number of binaries to simulate.
            Default is 1,000.
        n_workers (int): The number of workers to use for parallel processing.
            Default is 1.
            If -1, it will use all available cores.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        SimulationOutput: tuple(pd.DataFrame, BinaryGenerator)
            The dataframe contains the simulated binaries and their properties.
            The BinaryGenerator contains the black hole generator used for the simulation.
    """

    n_workers = get_n_workers(n_workers)

    bh1_generator = BlackHolePopulation.from_simulation_results(df=df_bh1_binaries)
    bh2_generator = (
        BlackHoleGenerator()
        if df_bh2_binaries is None
        else BlackHolePopulation.from_simulation_results(df=df_bh2_binaries)
    )
    binary_generator = BinaryGenerator(
        primary_black_hole_source=bh1_generator,
        secondary_black_hole_source=bh2_generator,
        is_aligned_spin=False,
    )
    black_hole_mergers = simulate_black_hole_mergers(
        binary_generator, Fits.NRSUR7DQ4REMNANT, size, n_workers, random_state
    )
    df_binaries = convert_simulated_binaries_to_dataframe(black_hole_mergers)
    return df_binaries, binary_generator


def simulate_multi_generation_aligned_spin_binaries(
    df_bh1_binaries: pd.DataFrame,
    df_bh2_binaries: Optional[pd.DataFrame] = None,
    size: int = 1000,
    n_workers: int = 1,
    random_state: int = 42,
) -> PipelineOutput:
    """Simulate a population of multi-generation aligned spin binaries.

    Args:
        df_bh1_binaries (pd.DataFrame):
            DataFrame containing the first generation black hole binaries.
        df_bh2_binaries (Optional[pd.DataFrame]):
            DataFrame containing the second generation black hole binaries.
            Default is None, meaning the second black hole will be drawn from a default distribution.
        size (int): The number of binaries to simulate.
            Default is 1,000.
        n_workers (int): The number of workers to use for parallel processing.
            Default is 1.
            If -1, it will use all available cores.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        SimulationOutput: tuple(pd.DataFrame, BinaryGenerator)
            The dataframe contains the simulated binaries and their properties.
            The BinaryGenerator contains the black hole generator used for the simulation.
    """

    n_workers = get_n_workers(n_workers)

    bh1_generator = BlackHolePopulation.from_simulation_results(df=df_bh1_binaries)
    bh2_generator = (
        BlackHoleGenerator()
        if df_bh2_binaries is None
        else BlackHolePopulation.from_simulation_results(df=df_bh2_binaries)
    )
    binary_generator = BinaryGenerator(
        primary_black_hole_source=bh1_generator,
        secondary_black_hole_source=bh2_generator,
        is_aligned_spin=True,
    )
    black_hole_mergers = simulate_black_hole_mergers(
        binary_generator, Fits.NRSUR3DQ8REMNANT, size, n_workers, random_state
    )
    df_binaries = convert_simulated_binaries_to_dataframe(black_hole_mergers)
    return df_binaries, binary_generator
