import numpy as np

from archeo.data_structures.physics.binary import Binary
from archeo.data_structures.physics.black_hole import BlackHole
from archeo.simulation.simulate_merger import _simulate_black_hole_merger, simulate_black_hole_mergers


class DummyFits:
    def vf(self, q, s1, s2):
        return np.array([0.001, 0.0, 0.0]), 0.0

    def chif(self, q, s1, s2):
        return np.array([0.0, 0.0, 0.5]), 0.0

    def mf(self, q, s1, s2):
        return 0.95, 0.0


class DummyFitsEnum:
    def load(self):
        return DummyFits()


class DummyBinaryGenerator:
    def draw(self, size=1):
        bhs = []
        for _ in range(size):
            p = BlackHole(mass=40.0, spin_magnitude=0.3, spin_vector=(0.0, 0.0, 0.3), speed=0.0)
            s = BlackHole(mass=20.0, spin_magnitude=0.2, spin_vector=(0.0, 0.0, 0.2), speed=0.0)
            bhs.append(Binary(primary_black_hole=p, secondary_black_hole=s))
        return bhs


def test_simulate_black_hole_merger_core_values():
    binary = DummyBinaryGenerator().draw(size=1)[0]
    remnant = _simulate_black_hole_merger(binary, DummyFits())
    assert remnant.mass > 0
    assert np.isclose(remnant.spin_magnitude, 0.5)


def test_simulate_black_hole_mergers_single_worker():
    out = simulate_black_hole_mergers(
        binary_generator=DummyBinaryGenerator(),
        fits=DummyFitsEnum(),
        size=5,
        n_workers=1,
        random_state=123,
    )
    assert len(out) == 5
