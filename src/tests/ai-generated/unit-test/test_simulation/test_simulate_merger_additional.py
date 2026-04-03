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
        out = []
        for _ in range(size):
            p = BlackHole(mass=40.0, spin_magnitude=0.4, spin_vector=(0.0, 0.0, 0.4), speed=0.0)
            s = BlackHole(mass=20.0, spin_magnitude=0.3, spin_vector=(0.0, 0.0, 0.3), speed=0.0)
            out.append(Binary(primary_black_hole=p, secondary_black_hole=s))
        return out


def test_simulate_black_hole_merger_core():
    binary = DummyBinaryGenerator().draw(size=1)[0]
    rem = _simulate_black_hole_merger(binary, DummyFits())
    assert rem.mass > 0
    assert np.isclose(rem.spin_magnitude, 0.5)
    assert rem.speed > 0


def test_simulate_black_hole_mergers_single_worker():
    out = simulate_black_hole_mergers(
        binary_generator=DummyBinaryGenerator(),
        fits=DummyFitsEnum(),
        size=7,
        n_workers=1,
        random_state=42,
    )
    assert len(out) == 7


def test_simulate_black_hole_mergers_parallel_remainder(monkeypatch):
    monkeypatch.setattr(
        "archeo.simulation.simulate_merger.multiprocess_run",
        lambda func, input_kwargs, n_processes: [func(**kwargs) for kwargs in input_kwargs],
    )
    out = simulate_black_hole_mergers(
        binary_generator=DummyBinaryGenerator(),
        fits=DummyFitsEnum(),
        size=10,  # with n_workers=3 -> chunk 3 each => 9 + remainder 1
        n_workers=3,
        random_state=42,
    )
    assert len(out) == 10
