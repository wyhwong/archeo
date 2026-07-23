import numpy as np

from archeo.data_structures.physics.binary import Binary
from archeo.data_structures.physics.black_hole import BlackHole
from archeo.simulation.simulate_merger import simulate_black_hole_mergers


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
    def draw(self, size=1, random_state=None):  # pylint: disable=unused-argument
        out = []
        for _ in range(size):
            p = BlackHole(mass=40.0, spin_magnitude=0.3, spin_vector=(0.0, 0.0, 0.3), speed=0.0)
            s = BlackHole(mass=20.0, spin_magnitude=0.2, spin_vector=(0.0, 0.0, 0.2), speed=0.0)
            out.append(Binary(primary_black_hole=p, secondary_black_hole=s))
        return out


def test_simulate_black_hole_mergers_workers_exceed_size_hits_remainder(monkeypatch):
    calls = {"chunk_sizes": None}

    def fake_multiprocess_run(func, input_kwargs, n_processes):
        calls["chunk_sizes"] = [kw["size"] for kw in input_kwargs]
        # emulate process pool behavior deterministically
        return [func(**kw) for kw in input_kwargs]

    monkeypatch.setattr("archeo.simulation.simulate_merger.multiprocess_run", fake_multiprocess_run)

    out = simulate_black_hole_mergers(
        binary_generator=DummyBinaryGenerator(),
        fits=DummyFitsEnum(),
        size=2,
        n_workers=5,  # chunk_size = 0, remainder branch must fill all samples
        random_state=123,
    )

    assert calls["chunk_sizes"] == [1, 1]
    assert len(out) == 2
