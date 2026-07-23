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
    random_state: Optional[int] = None,
) -> PipelineOutput:
    """Simulate multi-generation binaries with precession spins.

    Args:
        df_bh1_binaries (pd.DataFrame): Source population for the primary black hole.
        df_bh2_binaries (Optional[pd.DataFrame]): Optional source population for the
            secondary black hole. If ``None``, a default generator is used.
        size (int): Number of binaries to simulate.
        n_workers (int): Number of workers. Use `-1` for all available cores.
        random_state (int): Seed for reproducibility.

    Returns:
        PipelineOutput: Tuple containing:
            - pd.DataFrame: Simulated binary/remnant records.
            - BinaryGenerator: Generator used for simulation.
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
        binary_generator=binary_generator,
        fits=Fits.NRSUR7DQ4REMNANT,
        size=size,
        n_workers=n_workers,
        random_state=random_state,
    )
    df_binaries = convert_simulated_binaries_to_dataframe(black_hole_mergers)
    return df_binaries, binary_generator


def simulate_multi_generation_aligned_spin_binaries(
    df_bh1_binaries: pd.DataFrame,
    df_bh2_binaries: Optional[pd.DataFrame] = None,
    size: int = 1000,
    n_workers: int = 1,
    random_state: Optional[int] = None,
) -> PipelineOutput:
    """Simulate multi-generation binaries with aligned spins.

    Args:
        df_bh1_binaries (pd.DataFrame): Source population for the primary black hole.
        df_bh2_binaries (Optional[pd.DataFrame]): Optional source population for the
            secondary black hole. If ``None``, a default generator is used.
        size (int): Number of binaries to simulate.
        n_workers (int): Number of workers. Use `-1` for all available cores.
        random_state (int): Seed for reproducibility.

    Returns:
        PipelineOutput: Tuple containing:
            - pd.DataFrame: Simulated binary/remnant records.
            - BinaryGenerator: Generator used for simulation.
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
        binary_generator=binary_generator,
        fits=Fits.NRSUR3DQ8REMNANT,
        size=size,
        n_workers=n_workers,
        random_state=random_state,
    )
    df_binaries = convert_simulated_binaries_to_dataframe(black_hole_mergers)
    return df_binaries, binary_generator
