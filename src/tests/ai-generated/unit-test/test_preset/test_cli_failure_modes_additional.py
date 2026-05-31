import pandas as pd
from click.testing import CliRunner

from archeo.preset.cli import simulation_cli


def test_cli_simulate_agnostic_fails_when_output_dir_does_not_exist(monkeypatch, tmp_path):
    class DummyGenerator:
        def model_dump(self):
            return {"ok": True}

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

    monkeypatch.setattr(
        "archeo.preset.cli.simulate_agnostic_precession_spin_binaries",
        lambda size, n_workers: (tiny_df, DummyGenerator()),
    )

    missing_dir = tmp_path / "does_not_exist" / "nested"
    runner = CliRunner()
    result = runner.invoke(
        simulation_cli,
        [
            "simulate-agnostic-black-hole-population",
            "-n",
            "1",
            "-np",
            "1",
            "-o",
            str(missing_dir),
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, FileNotFoundError)


def test_cli_visualize_black_hole_population_rejects_unsupported_extension(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("x", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        simulation_cli,
        ["visualize-black-hole-population", "-f", str(bad), "-o", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
