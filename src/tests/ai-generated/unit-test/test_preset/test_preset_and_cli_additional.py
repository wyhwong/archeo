import pandas as pd
import pytest
from click.testing import CliRunner

from archeo.preset.cli import simulation_cli
from archeo.preset.forward.compute_bayes_factor_curve import compute_bayes_factor_curve_over_escape_velocity
from archeo.preset.simulation import get_binary_generation_pipeline


def test_get_binary_generation_pipeline_invalid():
    with pytest.raises(ValueError, match="Invalid binary generation pipeline name"):
        get_binary_generation_pipeline("does_not_exist")


def test_compute_bayes_factor_curve_over_escape_velocity_monkeypatched(monkeypatch):
    class DummyCandidatePrior:
        def __init__(self, df_bh1, df_bh2):
            self.df_bh1 = df_bh1
            self.df_bh2 = df_bh2

    class DummyCurve:
        def __init__(self):
            self.metadata = type(
                "M",
                (),
                {
                    "reference_candidate_name": "original",
                    "reference_bayes_factor": 1.0,
                    "binsize_spin": 0.1,
                    "binsize_mass": 2.0,
                },
            )()

        def get_bayes_factor_over_escape_velocity(self, prior, posterior, candidate_prior, n_workers):
            from archeo.data_structures.bayesian.bayes_factor import BayesFactor

            return {50.0: BayesFactor(samples=[1.0, 1.1, 0.9])}

    monkeypatch.setattr(
        "archeo.preset.forward.compute_bayes_factor_curve.CandidatePrior",
        DummyCandidatePrior,
    )
    monkeypatch.setattr(
        "archeo.preset.forward.compute_bayes_factor_curve.BayesFactorCurve",
        DummyCurve,
    )

    df_prior = pd.DataFrame({"m_1": [30.0], "a_1": [0.3]})
    df_post = pd.DataFrame({"m_1": [32.0], "a_1": [0.4]})
    df_bh1 = pd.DataFrame({"v_esc": [10.0], "m_1": [30.0]})
    df_bh2 = pd.DataFrame({"v_esc": [10.0], "m_2": [20.0]})

    out = compute_bayes_factor_curve_over_escape_velocity(df_prior, df_post, df_bh1, df_bh2, n_workers=1)
    assert "v_esc" in out.columns
    assert len(out) == 1


def test_cli_parquet_fallback_to_csv(monkeypatch, tmp_path):
    # lightweight simulation output (avoid heavy surfinBH call)
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
        "archeo.preset.cli.simulate_agnostic_aligned_spin_binaries",
        lambda size, n_workers: (tiny_df, DummyGenerator()),
    )
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, *args, **kwargs: (_ for _ in ()).throw(ImportError()))

    runner = CliRunner()
    result = runner.invoke(
        simulation_cli,
        [
            "simulate-agnostic-black-hole-population",
            "--aligned-spin",
            "-n",
            "1",
            "-np",
            "1",
            "-o",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "simulated_binaries.csv").exists()
    assert (tmp_path / "binary_generator_config.json").exists()
