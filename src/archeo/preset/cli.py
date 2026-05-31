import json

import click

from archeo.preset.simulation.agnostic import (
    simulate_agnostic_aligned_spin_binaries,
    simulate_agnostic_precession_spin_binaries,
)
from archeo.preset.simulation.second_generation import (
    simulate_second_generation_aligned_spin_binaries,
    simulate_second_generation_precession_spin_binaries,
)
from archeo.utils.fs import load_dataframe
from archeo.utils.parallel import get_n_workers
from archeo.visualization import visualize_prior_distribution


@click.group()
def simulation_cli():
    """Create the root command group for simulation-related CLI commands.

    Returns:
        None
    """


@simulation_cli.command()
@click.option("-n", "--size", default=500000, help="Number of black holes to simulate.")
@click.option("-np", "--n-workers", default=-1, help="Number of cores to use for simulation.")
@click.option("-o", "--output-dir", default=".", help="Directory to save the generated data.")
@click.option("-as", "--aligned-spin", is_flag=True, help="Toggle to simulate aligned spin binaries.")
def simulate_second_generation_black_hole_population(size: int, n_workers: int, output_dir: str, aligned_spin: bool):
    """Simulate and persist a second-generation black-hole binary population.

    This command generates binaries using either aligned-spin or precession-spin
    configuration, saves generator configuration as JSON, and writes samples as
    parquet when available (falls back to CSV otherwise).

    Args:
        size (int): Number of binaries to generate.
        n_workers (int): Number of worker processes. Use `-1` to use all cores.
        output_dir (str): Output directory for generated artifacts.
        aligned_spin (bool): If ``True``, generate aligned-spin binaries; otherwise,
            generate precession-spin binaries.

    Returns:
        None

    Command example:
    >> python -m archeo simulate-second-generation-black-hole-population --aligned-spin
    """

    n_workers = get_n_workers(n_workers)
    spin_desc = "aligned" if aligned_spin else "precession"
    click.echo(
        f"Generating {size} second generation black hole binaries with "
        f"{spin_desc} spin configuration "
        f"using {n_workers} workers..."
    )

    if aligned_spin:
        df_binaries, binary_generator = simulate_second_generation_aligned_spin_binaries(size=size, n_workers=n_workers)
    else:
        df_binaries, binary_generator = simulate_second_generation_precession_spin_binaries(
            size=size, n_workers=n_workers
        )

    with open(f"{output_dir}/binary_generator_config.json", "w", encoding="utf-8") as fp:
        json.dump(binary_generator.model_dump(), fp, indent=4)

    try:
        df_binaries.to_parquet(f"{output_dir}/simulated_binaries.parquet")
        click.echo(f"Prior saved as parquet: {output_dir}/simulated_binaries.parquet")
    except ImportError:
        click.echo("Failed to save prior as parquet, please install pyarrow if you want to use this feature.")
        df_binaries.to_csv(f"{output_dir}/simulated_binaries.csv", index=False)
        click.echo(f"Prior saved as CSV: {output_dir}/simulated_binaries.csv")


@simulation_cli.command()
@click.option("-n", "--size", default=500000, help="Number of black holes to simulate.")
@click.option("-np", "--n-workers", default=-1, help="Number of cores to use for simulation.")
@click.option("-o", "--output-dir", default=".", help="Directory to save the generated data.")
@click.option("-as", "--aligned-spin", is_flag=True, help="Toggle to simulate aligned spin binaries.")
def simulate_agnostic_black_hole_population(size: int, n_workers: int, output_dir: str, aligned_spin: bool):
    """Simulate and persist an agnostic black-hole binary population.

    This command generates binaries with broad agnostic priors using either
    aligned-spin or precession-spin configuration, saves generator configuration
    as JSON, and writes samples as parquet when available (falls back to CSV).

    Args:
        size (int): Number of binaries to generate.
        n_workers (int): Number of worker processes. Use `-1` to use all cores.
        output_dir (str): Output directory for generated artifacts.
        aligned_spin (bool): If ``True``, generate aligned-spin binaries; otherwise,
            generate precession-spin binaries.

    Returns:
        None

    Command example:
    >> python -m archeo simulate-agnostic-black-hole-population --aligned-spin
    """

    n_workers = get_n_workers(n_workers)
    spin_desc = "aligned" if aligned_spin else "precession"
    click.echo(
        f"Generating {size} agnostic black hole binaries with "
        f"{spin_desc} spin configuration "
        f"using {n_workers} workers..."
    )

    if aligned_spin:
        df_binaries, binary_generator = simulate_agnostic_aligned_spin_binaries(size=size, n_workers=n_workers)
    else:
        df_binaries, binary_generator = simulate_agnostic_precession_spin_binaries(size=size, n_workers=n_workers)

    with open(f"{output_dir}/binary_generator_config.json", "w", encoding="utf-8") as fp:
        json.dump(binary_generator.model_dump(), fp, indent=4)

    try:
        df_binaries.to_parquet(f"{output_dir}/simulated_binaries.parquet")
        click.echo(f"Prior saved as parquet: {output_dir}/simulated_binaries.parquet")
    except ImportError:
        click.echo("Failed to save prior as parquet, please install pyarrow if you want to use this feature.")
        df_binaries.to_csv(f"{output_dir}/simulated_binaries.csv", index=False)
        click.echo(f"Prior saved as CSV: {output_dir}/simulated_binaries.csv")


@simulation_cli.command()
@click.option("-f", "--filepath", required=True, help="Path to the binary data.")
@click.option("-o", "--output-dir", default=".", help="Directory to save visualizations.")
def visualize_black_hole_population(filepath: str, output_dir: str):
    """Generate standard prior visualizations for a stored binary population.

    Args:
        filepath (str): Path to an input dataframe file.
        output_dir (str): Directory where visualizations are written.

    Returns:
        None

    Command example:
    >> python -m archeo visualize-black-hole-population --filepath ./simulated_binaries.parquet
    """

    click.echo(f"Generating visualizations for black hole population at {filepath}")

    df_binaries = load_dataframe(filepath)
    visualize_prior_distribution(df_binaries, output_dir=output_dir)
