import os
from unittest.mock import Mock

import pandas as pd
from click.testing import CliRunner

from archeo.preset.cli import simulation_cli


def _fake_binary_generator():
    generator = Mock()
    generator.model_dump.return_value = {
        "mock": True,
        "is_aligned_spin": True,
    }
    return generator


def _fake_binaries_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "m_1": [30.0, 35.0],
            "a_1": [0.1, 0.2],
            "a_1x": [0.0, 0.0],
            "a_1y": [0.0, 0.0],
            "a_1z": [0.1, -0.2],
            "v_1": [0.0, 0.0],
            "m_2": [20.0, 25.0],
            "a_2": [0.3, 0.4],
            "a_2x": [0.0, 0.0],
            "a_2y": [0.0, 0.0],
            "a_2z": [0.3, -0.4],
            "v_2": [0.0, 0.0],
            "m_f": [48.0, 57.0],
            "a_f": [0.7, 0.8],
            "k_f": [100.0, 200.0],
            "chi_eff": [0.15, -0.25],
            "chi_p": [0.0, 0.0],
            "q": [1.5, 1.4],
        }
    )


def test_2g_bh_population_simulation_command(tmp_path, monkeypatch):
    output_dir = str(tmp_path)

    fake_df = _fake_binaries_dataframe()
    fake_generator = _fake_binary_generator()

    mock_simulate = Mock(return_value=(fake_df, fake_generator))

    monkeypatch.setattr(
        "archeo.preset.cli.simulate_second_generation_aligned_spin_binaries",
        mock_simulate,
    )

    runner = CliRunner()
    result = runner.invoke(
        simulation_cli,
        [
            "simulate-second-generation-black-hole-population",
            "--aligned-spin",
            "-n",
            "1000",
            "-np",
            "1",
            "-o",
            output_dir,
        ],
    )

    assert result.exit_code == 0, result.output

    assert (
        "Generating 1000 second generation black hole binaries with " "aligned spin configuration using 1 workers..."
    ) in result.output

    mock_simulate.assert_called_once_with(size=1000, n_workers=1)

    assert os.path.exists(f"{output_dir}/binary_generator_config.json")
    assert os.path.exists(f"{output_dir}/simulated_binaries.parquet") or os.path.exists(
        f"{output_dir}/simulated_binaries.csv"
    )
