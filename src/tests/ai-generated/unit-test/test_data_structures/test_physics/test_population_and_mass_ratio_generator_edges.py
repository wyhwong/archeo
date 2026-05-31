import numpy as np
import pandas as pd

from archeo.data_structures.distribution import Uniform
from archeo.data_structures.math import Domain
from archeo.data_structures.physics.binary import MassRatioBasedBinaryGenerator
from archeo.data_structures.physics.black_hole import BlackHolePopulation


def test_black_hole_population_from_simulation_results_and_draw_with_replacement():
    df = pd.DataFrame(
        {
            "m_f": [40.0, 50.0],
            "a_f": [0.3, 0.7],
            "k_f": [100.0, 250.0],
        }
    )

    pop = BlackHolePopulation.from_simulation_results(
        df=df,
        phi_distribution=Uniform(low=0.0, high=0.0),
        theta_distribution=Uniform(low=0.0, high=0.0),
    )

    assert len(pop.black_holes) == 2
    assert [bh.mass for bh in pop.black_holes] == [40.0, 50.0]
    assert [bh.speed for bh in pop.black_holes] == [100.0, 250.0]
    # theta=0 and phi=0 => spin vector on +z only
    assert np.isclose(pop.black_holes[0].spin_vector[0], 0.0)
    assert np.isclose(pop.black_holes[0].spin_vector[1], 0.0)
    assert np.isclose(pop.black_holes[0].spin_vector[2], 0.3)

    drawn = pop.draw(size=10)
    assert len(drawn) == 10
    assert set((bh.mass, bh.speed) for bh in drawn).issubset({(40.0, 100.0), (50.0, 250.0)})


def test_mass_ratio_based_binary_generator_enforces_secondary_domain_and_aligned_spins():
    np.random.seed(7)

    gen = MassRatioBasedBinaryGenerator(
        mass_ratio_distribution=Uniform(low=1.0, high=6.0),
        primary_mass_distribution=Uniform(low=5.0, high=65.0),
        secondary_mass_domain=Domain(low=5.0, high=65.0),
        is_aligned_spin=True,
    )

    binaries = gen.draw(size=200)
    assert len(binaries) == 200

    for b in binaries:
        m1 = b.primary_black_hole.mass
        m2 = b.secondary_black_hole.mass
        assert m1 >= m2
        assert 5.0 <= m2 <= 65.0

        # aligned-spin projection should zero out x/y components
        a1x, a1y, a1z = b.primary_black_hole.spin_vector
        a2x, a2y, a2z = b.secondary_black_hole.spin_vector
        assert a1x == 0.0 and a1y == 0.0
        assert a2x == 0.0 and a2y == 0.0
        assert np.isclose(abs(a1z), b.primary_black_hole.spin_magnitude)
        assert np.isclose(abs(a2z), b.secondary_black_hole.spin_magnitude)
