import os

import pytest
from click.testing import CliRunner

from archeo.preset.cli import simulation_cli


@pytest.fixture(name="output_dir")
def get_output_dir() -> str:
    """Fixture that provides the output directory for the simulation commands."""

    return f"{os.path.dirname(os.path.dirname(__file__))}/test_data"


def test_2g_bh_population_simulation_command(output_dir: str):

    assert not os.path.exists(f"{output_dir}/simulated_binaries.parquet")
    assert not os.path.exists(f"{output_dir}/simulated_binaries.csv")
    assert not os.path.exists(f"{output_dir}/binary_generator_config.json")

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
    assert result.exit_code == 0
    assert (
        "Generating 1000 second generation black hole binaries with " "aligned spin configuration using 1 workers..."
    ) in result.output
    assert os.path.exists(f"{output_dir}/simulated_binaries.parquet") or os.path.exists(
        f"{output_dir}/simulated_binaries.csv"
    )
    assert os.path.exists(f"{output_dir}/binary_generator_config.json")

    _ = (
        os.remove(f"{output_dir}/simulated_binaries.parquet")
        if os.path.exists(f"{output_dir}/simulated_binaries.parquet")
        else os.remove(f"{output_dir}/simulated_binaries.csv")
    )
    os.remove(f"{output_dir}/binary_generator_config.json")

    assert not os.path.exists(f"{output_dir}/simulated_binaries.parquet")
    assert not os.path.exists(f"{output_dir}/simulated_binaries.csv")
    assert not os.path.exists(f"{output_dir}/binary_generator_config.json")


def test_agnostic_bh_population_simulation_command(output_dir: str):

    assert not os.path.exists(f"{output_dir}/simulated_binaries.parquet")
    assert not os.path.exists(f"{output_dir}/simulated_binaries.csv")
    assert not os.path.exists(f"{output_dir}/binary_generator_config.json")

    runner = CliRunner()
    result = runner.invoke(
        simulation_cli,
        [
            "simulate-agnostic-black-hole-population",
            "--aligned-spin",
            "-n",
            "1000",
            "-np",
            "1",
            "-o",
            output_dir,
        ],
    )
    assert result.exit_code == 0
    assert (
        "Generating 1000 agnostic black hole binaries with " "aligned spin configuration using 1 workers..."
    ) in result.output
    assert os.path.exists(f"{output_dir}/simulated_binaries.parquet") or os.path.exists(
        f"{output_dir}/simulated_binaries.csv"
    )
    assert os.path.exists(f"{output_dir}/binary_generator_config.json")

    _ = (
        os.remove(f"{output_dir}/simulated_binaries.parquet")
        if os.path.exists(f"{output_dir}/simulated_binaries.parquet")
        else os.remove(f"{output_dir}/simulated_binaries.csv")
    )
    os.remove(f"{output_dir}/binary_generator_config.json")

    assert not os.path.exists(f"{output_dir}/simulated_binaries.parquet")
    assert not os.path.exists(f"{output_dir}/simulated_binaries.csv")
    assert not os.path.exists(f"{output_dir}/binary_generator_config.json")
