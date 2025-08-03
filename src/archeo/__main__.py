import click

import archeo
from archeo.preset import PRIOR_STORE, get_prior_config


@click.group()
def cli():
    """Command line interface for archeo"""


@cli.command()
@click.option(
    "-n",
    "--name",
    default="default",
    help=f"Preset prior name, available values are {', '.join(PRIOR_STORE)}.",
)
@click.option(
    "-o",
    "--output-dir",
    default=".",
    help="Directory to save the generated prior configuration.",
)
def generate_preset_prior(name: str, output_dir: str):
    """Generate a preset prior."""

    click.echo(f"Generating preset prior: {name}")

    prior_config = get_prior_config(name)
    click.echo(f"Prior configuration: {prior_config}")

    output_path = f"{output_dir}/{name}_prior_config.yaml"
    prior_config.to_yaml(output_path)

    prior = archeo.Prior.from_config(prior_config)

    try:
        prior.to_parquet(f"{output_dir}/{name}_prior.parquet")
        click.echo(f"Prior saved as parquet: {output_dir}/{name}_prior.parquet")
    except ImportError:
        click.echo("Failed to save prior as parquet, please install pyarrow if you want to use this feature.")
        prior.to_csv(f"{output_dir}/{name}_prior.csv")
        click.echo(f"Prior saved as CSV: {output_dir}/{name}_prior.csv")


@cli.command()
@click.option(
    "-f",
    "--filepath",
    required=True,
    help="Path to the prior data.",
)
@click.option(
    "-o",
    "--output-dir",
    default=".",
    help="Directory to save the visualization output.",
)
def visualize_prior(filepath: str, output_dir: str):
    """Visualize the prior distribution."""

    prior = archeo.Prior.from_filepath(filepath)
    archeo.visualize_prior_distribution(prior, output_dir=output_dir)


if __name__ == "__main__":
    cli()
