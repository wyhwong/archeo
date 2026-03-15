import numpy as np

from archeo.constants.enum import Fits
from archeo.data_structures.physics.binary import Binaries, Binary, BinaryGenerator
from archeo.data_structures.physics.black_hole import BlackHole, BlackHoles
from archeo.utils.parallel import multithread_run


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
        speed=np.linalg.norm(birth_recoil_vec),
    )


def simulate_black_hole_mergers(
    binary_generator: BinaryGenerator,
    fits: Fits,
    size: int,
) -> tuple[Binaries, BlackHoles]:
    """Simulate black hole mergers.

    Args:
        binary_generator (BinaryGenerator): Binary generator to draw binaries from
        fits (Fits): surfinBH model to use for the simulation
        size (int): Number of mergers to simulate

    Returns:
        tuple[Binaries, BlackHoles]: Tuple of binaries and black holes resulting from the mergers
    """

    binaries = binary_generator.draw(size=size)
    loaded_fits = fits.load()

    remnants = multithread_run(
        _simulate_black_hole_merger, [{"binary": binary, "loaded_fits": loaded_fits} for binary in binaries]
    )
    return binaries, remnants
