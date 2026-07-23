from typing import Optional

import numpy as np

from archeo.constants.enum import Fits
from archeo.constants.physics import SPEED_OF_LIGHT
from archeo.data_structures.physics.binary import Binary, BinaryGenerator
from archeo.data_structures.physics.black_hole import BlackHole
from archeo.data_structures.physics.simulation import BlackHoleMergers
from archeo.utils.parallel import multiprocess_run, multithread_run


def _simulate_black_hole_merger(binary: Binary, loaded_fits) -> BlackHole:
    """Simulate a single binary merger and return remnant black-hole properties.

    Args:
        binary (Binary): Input binary system.
        loaded_fits: Preloaded surrogate-fit object implementing ``vf``, ``chif``,
            and ``mf``.

    Returns:
        BlackHole: Remnant black hole inferred from the surrogate model.
    """

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
    n_threads: int = 1,
    random_state: Optional[int] = None,
) -> BlackHoleMergers:
    """Simulate a batch of black-hole mergers in a single process.

    Args:
        binary_generator (BinaryGenerator): Binary source generator.
        fits (Fits): Surrogate model enum entry.
        size (int): Number of mergers to simulate.
        n_threads (int): Number of threads to use for simulation.
        random_state (int): Seed for reproducibility.

    Returns:
        BlackHoleMergers: List of ``(binary, remnant)`` tuples.
    """

    binaries = binary_generator.draw(size=size, random_state=random_state)
    loaded_fits = fits.load()

    remnants = multithread_run(
        func=_simulate_black_hole_merger,
        input_kwargs=[
            {
                "binary": binary,
                "loaded_fits": loaded_fits,
            }
            for binary in binaries
        ],
        n_threads=n_threads,
    )
    return list(zip(binaries, remnants))


def simulate_black_hole_mergers(
    binary_generator: BinaryGenerator,
    fits: Fits,
    size: int,
    n_workers: int = 1,
    n_threads_per_worker: int = 1,
    random_state: Optional[int] = None,
) -> BlackHoleMergers:
    """Simulate black-hole mergers with optional multiprocessing.

    Args:
        binary_generator (BinaryGenerator): Binary source generator.
        fits (Fits): Surrogate model enum entry.
        size (int): Number of mergers to simulate.
        n_workers (int): Number of worker processes.
        n_threads_per_worker (int): Number of threads to use for each worker process.
        random_state (Optional[int]): Base random seed.

    Returns:
        BlackHoleMergers: List of ``(binary, remnant)`` tuples.
    """

    if n_workers == 1:
        return _simulate_black_hole_mergers(
            binary_generator=binary_generator,
            fits=fits,
            size=size,
            random_state=random_state,
            n_threads=n_threads_per_worker,
        )

    # If n_workers > 1, we can parallelize the simulation by splitting the size into chunks
    n_workers = min(n_workers, size)
    chunk_sizes = [size // n_workers] * n_workers
    seed_sequence = np.random.SeedSequence(random_state)
    random_states = seed_sequence.generate_state(n_workers)
    for i in range(size % n_workers):
        chunk_sizes[i] += 1

    results = multiprocess_run(
        func=_simulate_black_hole_mergers,
        input_kwargs=[
            {
                "binary_generator": binary_generator,
                "fits": fits,
                "size": chunk_size,
                "random_state": _random_state,
                "n_threads": n_threads_per_worker,
            }
            for _random_state, chunk_size in zip(random_states, chunk_sizes)
        ],
        n_processes=n_workers,
    )
    # Combine the results from the different processes
    black_hole_mergers = sum(results, [])

    return black_hole_mergers
