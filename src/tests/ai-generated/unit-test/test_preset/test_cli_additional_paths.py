import tempfile
from unittest.mock import patch

import pandas as pd
from click.testing import CliRunner

from archeo.preset.cli import simulation_cli


def test_cli_simulate_second_generation_defaults_to_precession(monkeypatch, tmp_path):
    called = {"precession": 0, "aligned": 0}

    class DummyGenerator:
        def model_dump(self):
            return {"dummy": True}

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
        "archeo.preset.cli.simulate_second_generation_precession_spin_binaries",
        lambda size, n_workers: (
            called.__setitem__("precession", called["precession"] + 1) or (tiny_df, DummyGenerator())
        ),
    )
    monkeypatch.setattr(
        "archeo.preset.cli.simulate_second_generation_aligned_spin_binaries",
        lambda size, n_workers: (called.__setitem__("aligned", called["aligned"] + 1) or (tiny_df, DummyGenerator())),
    )
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, *args, **kwargs: None)

    runner = CliRunner()
    result = runner.invoke(
        simulation_cli,
        ["simulate-second-generation-black-hole-population", "-n", "1", "-np", "1", "-o", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "precession spin configuration" in result.output
    assert called["precession"] == 1
    assert called["aligned"] == 0
    assert (tmp_path / "binary_generator_config.json").exists()


def test_cli_visualize_command_calls_loader_and_visualizer(monkeypatch, tmp_path):
    called = {}

    fake_df = pd.DataFrame({"m_1": [1.0]})

    def fake_load_dataframe(fp):
        called.setdefault("filepath", fp)
        return fake_df

    def fake_visualize(df, output_dir):
        called.update({"visualized": True, "output_dir": output_dir, "rows": len(df)})

    with tempfile.TemporaryDirectory() as td:
        with (
            patch("archeo.preset.cli.load_dataframe", new=fake_load_dataframe),
            patch("archeo.preset.cli.visualize_prior_distribution", new=fake_visualize),
        ):

            runner = CliRunner()
            result = runner.invoke(
                simulation_cli,
                ["visualize-black-hole-population", "-f", "dummy.csv", "-o", td],
            )

    assert result.exit_code == 0
    assert called["filepath"] == "dummy.csv"
    assert called["visualized"] is True
    assert called["output_dir"] == td
    assert called["rows"] == 1


def test_cli_visualize_command_propagates_load_errors(monkeypatch):
    monkeypatch.setattr("archeo.preset.cli.load_dataframe", lambda fp: (_ for _ in ()).throw(ValueError("bad file")))

    runner = CliRunner()
    result = runner.invoke(
        simulation_cli,
        ["visualize-black-hole-population", "-f", "bad.ext", "-o", "."],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
