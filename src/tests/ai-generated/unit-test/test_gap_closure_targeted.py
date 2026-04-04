import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from click.testing import CliRunner

from archeo.bayesian.importance_sampling import ImportanceSamplingData as ISData
from archeo.bayesian.importance_sampling.bayes_factor_curve import BayesFactorCurve, CandidatePrior
from archeo.data_structures.bayesian.bayes_factor import BayesFactor
from archeo.data_structures.distribution import Uniform
from archeo.data_structures.math import Domain
from archeo.data_structures.physics.binary import BinaryGenerator
from archeo.data_structures.physics.black_hole import BlackHole, BlackHoleGenerator
from archeo.preset.cli import simulation_cli
from archeo.visualization.estimation import corner_estimates


def _df(n=200, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "m_1": rng.uniform(5, 65, n),
            "a_1": rng.uniform(0, 1, n),
        }
    )


def test_isdata_get_bayes_factor_dispatches_to_1d_branch(monkeypatch):
    data = ISData(
        prior_samples=_df(),
        posterior_samples=_df(seed=8),
        new_prior_samples=_df(seed=9),
        assume_parameter_independence=True,
    )

    monkeypatch.setattr(ISData, "get_bayes_factor_1d", lambda self, bootstrapping=False: 1.23)
    monkeypatch.setattr(
        ISData, "get_bayes_factor_dd", lambda self, bootstrapping=False: (_ for _ in ()).throw(RuntimeError)
    )

    assert data.get_bayes_factor() == 1.23


def test_isdata_get_bayes_factor_dispatches_to_dd_branch(monkeypatch):
    data = ISData(
        prior_samples=_df(),
        posterior_samples=_df(seed=8),
        new_prior_samples=_df(seed=9),
        assume_parameter_independence=False,
    )

    monkeypatch.setattr(ISData, "get_bayes_factor_dd", lambda self, bootstrapping=False: 0.77)
    monkeypatch.setattr(
        ISData, "get_bayes_factor_1d", lambda self, bootstrapping=False: (_ for _ in ()).throw(RuntimeError)
    )

    assert data.get_bayes_factor() == 0.77


def test_get_bayes_factor_1d_non_bootstrapping_path_runs():
    data = ISData(
        prior_samples=_df(),
        posterior_samples=_df(seed=10),
        new_prior_samples=_df(seed=11),
        assume_parameter_independence=True,
    )
    out = data.get_bayes_factor_1d(bootstrapping=False)
    assert np.isfinite(out)


def test_get_bayes_factor_dd_non_bootstrapping_path_runs():
    data = ISData(
        prior_samples=_df(),
        posterior_samples=_df(seed=10),
        new_prior_samples=_df(seed=11),
        assume_parameter_independence=False,
    )
    out = data.get_bayes_factor_dd(bootstrapping=False)
    assert np.isfinite(out)


def test_bayes_factor_curve_sample_bayes_factor_forces_parallel_sampling(monkeypatch):
    calls = {"n": None, "is_parallel": None}

    def fake_sample(self, n, is_parallel=False, n_threads=None):  # pylint: disable=unused-argument
        calls["n"] = n
        calls["is_parallel"] = is_parallel
        return BayesFactor(samples=[1.0, 1.0])

    monkeypatch.setattr(ISData, "sample_bayes_factor", fake_sample)

    curve = BayesFactorCurve(n_bootstrapping=2, n_pts=2, log_scale=False)
    prior = pd.DataFrame({"m_1": [30.0, 31.0], "a_1": [0.2, 0.3]})
    post = pd.DataFrame({"m_1": [32.0, 33.0], "a_1": [0.4, 0.5]})
    cp = CandidatePrior(
        df_bh1=pd.DataFrame({"v_esc": [50.0, 100.0], "m_1": [30.0, 40.0]}),
        df_bh2=pd.DataFrame({"v_esc": [50.0, 100.0], "m_2": [20.0, 25.0]}),
    )

    out = curve._sample_bayes_factor(prior=prior, posterior=post, candidate_prior=cp, v_esc=80.0)
    assert isinstance(out, BayesFactor)
    assert calls["n"] == 2
    assert calls["is_parallel"] is True


def test_binary_generator_enforce_source_binding_can_skip_invalid_pairs():
    class ScriptedSource:
        def __init__(self, scripted_draws):
            self.scripted_draws = scripted_draws
            self.calls = 0

        def draw(self, size=1):  # pylint: disable=unused-argument
            out = self.scripted_draws[self.calls]
            self.calls += 1
            return out

    # first loop (size=2): one invalid pair p<s (should hit continue), one valid pair
    # second loop (remaining_size=1): one valid pair
    p1 = BlackHole(mass=10.0, spin_magnitude=0.2, spin_vector=(0.0, 0.0, 0.2), speed=0.0)
    p2 = BlackHole(mass=30.0, spin_magnitude=0.2, spin_vector=(0.0, 0.0, 0.2), speed=0.0)
    p3 = BlackHole(mass=35.0, spin_magnitude=0.2, spin_vector=(0.0, 0.0, 0.2), speed=0.0)

    s1 = BlackHole(mass=20.0, spin_magnitude=0.1, spin_vector=(0.0, 0.0, 0.1), speed=0.0)
    s2 = BlackHole(mass=10.0, spin_magnitude=0.1, spin_vector=(0.0, 0.0, 0.1), speed=0.0)
    s3 = BlackHole(mass=15.0, spin_magnitude=0.1, spin_vector=(0.0, 0.0, 0.1), speed=0.0)

    primary_source = ScriptedSource([[p1, p2], [p3]])
    secondary_source = ScriptedSource([[s1, s2], [s3]])

    # model_construct bypasses strict type validation; good for targeted branch test
    gen = BinaryGenerator.model_construct(
        primary_black_hole_source=primary_source,
        secondary_black_hole_source=secondary_source,
        mass_ratio_domain=Domain(low=1.0, high=10.0),
        is_aligned_spin=False,
        enforce_source_binding=True,
    )

    binaries = gen.draw(size=2)
    assert len(binaries) == 2
    assert primary_source.calls == 2
    assert secondary_source.calls == 2
    assert all(b.primary_black_hole.mass >= b.secondary_black_hole.mass for b in binaries)


def test_black_hole_generator_validators_accept_valid_ranges():
    g = BlackHoleGenerator(
        spin_magnitude_distribution=Uniform(low=0.0, high=1.0),
        phi_distribution=Uniform(low=0.0, high=2 * np.pi),
        theta_distribution=Uniform(low=0.0, high=np.pi),
    )
    bhs = g.draw(size=5)
    assert len(bhs) == 5


def test_cli_second_generation_precession_parquet_fallback_to_csv(monkeypatch, tmp_path):
    tiny_df = pd.DataFrame(
        {
            "m_1": [30.0],
            "m_2": [20.0],
            "m_f": [47.0],
            "a_f": [0.7],
            "k_f": [100.0],
            "chi_eff": [0.1],
            "chi_p": [0.2],
            "q": [1.5],
        }
    )

    class DummyGenerator:
        def model_dump(self):
            return {"ok": True}

    monkeypatch.setattr(
        "archeo.preset.cli.simulate_second_generation_precession_spin_binaries",
        lambda size, n_workers: (tiny_df, DummyGenerator()),
    )
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, *args, **kwargs: (_ for _ in ()).throw(ImportError()))

    runner = CliRunner()
    result = runner.invoke(
        simulation_cli,
        [
            "simulate-second-generation-black-hole-population",
            "-n",
            "1",
            "-np",
            "1",
            "-o",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "precession spin configuration" in result.output
    assert (tmp_path / "simulated_binaries.csv").exists()
    assert (tmp_path / "binary_generator_config.json").exists()


def test_corner_estimates_warns_when_chi_p_mixed_availability(monkeypatch):
    calls = {"warn": 0, "corner": 0}

    monkeypatch.setattr(
        "archeo.visualization.estimation.LOGGER.warning",
        lambda *args, **kwargs: calls.__setitem__("warn", calls["warn"] + 1),
    )

    # Stub corner plotting so this test only checks orchestration/branching
    def _fake_corner(*args, **kwargs):
        calls["corner"] += 1
        return kwargs.get("fig", plt.figure())

    monkeypatch.setattr("archeo.visualization.estimation.corner.corner", _fake_corner)

    # avoid file I/O
    monkeypatch.setattr(
        "archeo.visualization.estimation.base.savefig_and_close",
        lambda *args, **kwargs: None,
    )

    base_cols = {
        "m_1": [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0],
        "m_2": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0],
        "m_f": [48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
        "k_f": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0],
        "a_f": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
        "chi_eff": [0.10, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24],
    }

    df_no_prec = pd.DataFrame({**base_cols, "chi_p": [0.0] * 8})
    df_with_prec = pd.DataFrame({**base_cols, "chi_p": [0.2, 0.3, 0.25, 0.35, 0.4, 0.45, 0.5, 0.55]})

    corner_estimates(
        {"no_prec": df_no_prec, "with_prec": df_with_prec},
        nbins=2,
        output_dir=None,
        close=True,
        fmt="png",
    )

    assert calls["warn"] == 1
    assert calls["corner"] > 0
