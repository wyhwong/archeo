import pandas as pd

from archeo.constants.enum import Fits
from archeo.preset.simulation.n_generation import (
    simulate_multi_generation_aligned_spin_binaries,
    simulate_multi_generation_precession_spin_binaries,
)
from archeo.preset.simulation.second_generation import (
    simulate_second_generation_aligned_spin_binaries,
    simulate_second_generation_precession_spin_binaries,
)


def test_second_generation_pipeline_selects_expected_fits_and_spin(monkeypatch):
    called = {}

    def fake_get_n_workers(n_workers):
        called["n_workers_in"] = n_workers
        return 3

    def fake_simulate(binary_generator, fits, size, n_workers, random_state):
        called["fits"] = fits
        called["size"] = size
        called["n_workers"] = n_workers
        called["aligned"] = binary_generator.is_aligned_spin
        return [("dummy_binary", "dummy_remnant")]

    monkeypatch.setattr("archeo.preset.simulation.second_generation.get_n_workers", fake_get_n_workers)
    monkeypatch.setattr("archeo.preset.simulation.second_generation.simulate_black_hole_mergers", fake_simulate)
    monkeypatch.setattr(
        "archeo.preset.simulation.second_generation.convert_simulated_binaries_to_dataframe",
        lambda mergers: pd.DataFrame({"ok": [len(mergers)]}),
    )

    df_a, gen_a = simulate_second_generation_aligned_spin_binaries(size=7, n_workers=-1, random_state=11)
    assert called["n_workers_in"] == -1
    assert called["n_workers"] == 3
    assert called["fits"] == Fits.NRSUR3DQ8REMNANT
    assert called["aligned"] is True
    assert len(df_a) == 1
    assert gen_a.is_aligned_spin is True

    df_p, gen_p = simulate_second_generation_precession_spin_binaries(size=9, n_workers=2, random_state=22)
    assert called["fits"] == Fits.NRSUR7DQ4REMNANT
    assert called["aligned"] is False
    assert len(df_p) == 1
    assert gen_p.is_aligned_spin is False


def test_multi_generation_pipeline_bh2_source_default_and_override(monkeypatch):
    called = {}

    def fake_simulate(binary_generator, fits, size, n_workers, random_state):
        called["secondary_source_type"] = type(binary_generator.secondary_black_hole_source).__name__
        called["fits"] = fits
        called["aligned"] = binary_generator.is_aligned_spin
        return [("dummy_binary", "dummy_remnant")]

    monkeypatch.setattr("archeo.preset.simulation.n_generation.simulate_black_hole_mergers", fake_simulate)
    monkeypatch.setattr(
        "archeo.preset.simulation.n_generation.convert_simulated_binaries_to_dataframe",
        lambda mergers: pd.DataFrame({"ok": [len(mergers)]}),
    )

    df_seed = pd.DataFrame({"m_f": [40.0, 50.0], "a_f": [0.3, 0.7], "k_f": [100.0, 200.0]})

    # default bh2 source => BlackHoleGenerator
    _, gen_default = simulate_multi_generation_precession_spin_binaries(
        df_bh1_binaries=df_seed, df_bh2_binaries=None, size=5, n_workers=1, random_state=1
    )
    assert called["secondary_source_type"] == "BlackHoleGenerator"
    assert called["fits"] == Fits.NRSUR7DQ4REMNANT
    assert gen_default.is_aligned_spin is False

    # overridden bh2 source => BlackHolePopulation
    _, gen_override = simulate_multi_generation_aligned_spin_binaries(
        df_bh1_binaries=df_seed, df_bh2_binaries=df_seed, size=5, n_workers=1, random_state=1
    )
    assert called["secondary_source_type"] == "BlackHolePopulation"
    assert called["fits"] == Fits.NRSUR3DQ8REMNANT
    assert gen_override.is_aligned_spin is True
