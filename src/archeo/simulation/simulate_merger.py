import numpy as np

from archeo.constants.enum import Fits
from archeo.constants.physics import SPEED_OF_LIGHT
from archeo.data_structures.physics.binary import Binary, BinaryGenerator
from archeo.data_structures.physics.black_hole import BlackHole
from archeo.data_structures.physics.simulation import BlackHoleMergers
from archeo.utils.parallel import multiprocess_run, multithread_run


def _simulate_black_hole_merger(binary: Binary, loaded_fits) -> BlackHole:
    q = binary.primary_black_hole.mass / binary.secondary_black_hole.mass
    birth_recoil_vec, birth_recoil_vec_err = loaded_fits.vf(  # pylint: disable=unused-variable
        q,
        binary.primary_black_hole.spin_vector,
        binary.secondary_black_hole.spin_vector,
    )
    spin_vec, spin_vec_err = loaded_fits.chif(  # pylint: disable=unused-variable
        q,
        binary.primary_black_hole.spin_vector,
        binary.secondary_black_hole.spin_vector,
    )
    m_retained, m_retained_err = loaded_fits.mf(  # pylint: disable=unused-variable
        q,
        binary.primary_black_hole.spin_vector,
        binary.secondary_black_hole.spin_vector,
    )
    return BlackHole(
        mass=m_retained * (binary.primary_black_hole.mass + binary.secondary_black_hole.mass),
        spin_magnitude=np.linalg.norm(spin_vec),
        spin_vector=spin_vec,
        speed=np.linalg.norm(birth_recoil_vec) * SPEED_OF_LIGHT,
    )


def _simulate_black_hole_mergers(
    binary_generator: BinaryGenerator,
    fits: Fits,
    size: int,
    random_state: int = 42,
) -> BlackHoleMergers:
    """Simulate black hole mergers.

    Args:
        binary_generator (BinaryGenerator): Binary generator to draw binaries from
        fits (Fits): surfinBH model to use for the simulation
        size (int): Number of mergers to simulate
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        BlackHoleMergers: List of tuples containing the binaries and their resulting black holes
    """

    np.random.seed(random_state)

    binaries = binary_generator.draw(size=size)
    loaded_fits = fits.load()

    remnants = multithread_run(
        _simulate_black_hole_merger, [{"binary": binary, "loaded_fits": loaded_fits} for binary in binaries]
    )
    return list(zip(binaries, remnants))


def simulate_black_hole_mergers(
    binary_generator: BinaryGenerator,
    fits: Fits,
    size: int,
    n_workers: int = 1,
    random_state: int = 42,
) -> BlackHoleMergers:
    """Simulate black hole mergers.

    Args:
        binary_generator (BinaryGenerator): Binary generator to draw binaries from
        fits (Fits): surfinBH model to use for the simulation
        size (int): Number of mergers to simulate
        n_workers (int): Number of worker processes to use for parallelization. Default is 1.
        random_state (int): Random state for reproducibility. Default is 42.
    Returns:
        BlackHoleMergers: List of tuples containing the binaries and their resulting black holes
    """

    if n_workers == 1:
        return _simulate_black_hole_mergers(binary_generator, fits, size, random_state)

    # If n_workers > 1, we can parallelize the simulation by splitting the size into chunks
    chunk_size = size // n_workers
    results = multiprocess_run(
        func=_simulate_black_hole_mergers,
        input_kwargs=[
            {"binary_generator": binary_generator, "fits": fits, "size": chunk_size, "random_state": random_state + i}
            for i in range(n_workers)
        ],
        n_processes=n_workers,
    )
    # Combine the results from the different processes
    black_hole_mergers = sum(results, [])

    if len(black_hole_mergers) < size:
        # If there are any remaining mergers to simulate (due to rounding), simulate them in the main process
        remaining_size = size - len(black_hole_mergers)
        remaining_bh_mergers = _simulate_black_hole_mergers(
            binary_generator, fits, remaining_size, random_state=random_state + n_workers
        )
        black_hole_mergers.extend(remaining_bh_mergers)

    return black_hole_mergers
