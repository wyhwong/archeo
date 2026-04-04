import pandas as pd

from archeo.constants.enum import Fits
from archeo.preset.simulation.agnostic import simulate_agnostic_precession_spin_binaries


def test_agnostic_precession_pipeline_uses_precession_fit(monkeypatch):
    called = {}

    def fake_get_n_workers(n):
        called["n_workers_in"] = n
        return 2

    def fake_simulate(binary_generator, fits, size, n_workers, random_state):
        called["fits"] = fits
        called["aligned"] = binary_generator.is_aligned_spin
        called["size"] = size
        called["n_workers"] = n_workers
        return [("b", "r")]

    monkeypatch.setattr("archeo.preset.simulation.agnostic.get_n_workers", fake_get_n_workers)
    monkeypatch.setattr("archeo.preset.simulation.agnostic.simulate_black_hole_mergers", fake_simulate)
    monkeypatch.setattr(
        "archeo.preset.simulation.agnostic.convert_simulated_binaries_to_dataframe",
        lambda mergers: pd.DataFrame({"n": [len(mergers)]}),
    )

    df, generator = simulate_agnostic_precession_spin_binaries(size=7, n_workers=-1, random_state=123)

    assert called["n_workers_in"] == -1
    assert called["n_workers"] == 2
    assert called["fits"] == Fits.NRSUR7DQ4REMNANT
    assert called["aligned"] is False
    assert len(df) == 1
    assert generator.is_aligned_spin is False
