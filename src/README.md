# ARCHEO (V2)

[![documentation](https://img.shields.io/badge/docs-archeo-blue.svg)](https://wyhwong.github.io/archeo/)
[![github](https://img.shields.io/badge/GitHub-archeo-blue.svg)](https://github.com/wyhwong/archeo)
[![PyPI version](https://badge.fury.io/py/archeo.svg)](https://pypi.org/project/archeo/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17632389.svg)](https://doi.org/10.5281/zenodo.14306853)
[![Downloads](https://img.shields.io/pepy/dt/archeo)](https://github.com/wyhwong/archeo)
[![Python version](https://img.shields.io/pypi/pyversions/archeo)](https://pypi.org/project/archeo/)
[![license](https://img.shields.io/badge/license-MIT-orange.svg)](https://github.com/wyhwong/archeo/blob/main/LICENSE)
[![CI](https://github.com/wyhwong/archeo/actions/workflows/ci.yml/badge.svg)](https://github.com/wyhwong/archeo/actions/workflows/ci.yml/)

Archeo is a Python package designed to infer the natal kick, ancestral masses,
and spins of black holes in the Pair-instability Supernova (PISN) gap,
with a particular focus on hierarchical black hole formation.

## Basic Usage with Command Line Interface (CLI)

We provide a command line interface (CLI) for archeo, which allows users to generate preset black hole binaries and visualize black hole population properties easily.

```
> python -m archeo --help

Usage: python -m archeo [OPTIONS] COMMAND [ARGS]...

  Command line interface for archeo

Options:
  --help  Show this message and exit.

Commands:
  - simulate-agnostic-black-hole-population
  - simulate-second-generation-black-hole-population
  - visualize-black-hole-population
```

In the following,
we will introduce the available commands in the CLI.

### Simulate a black hole remnant population with agnostic assumptions/second generation mergers

We provide a command to generate black hole binaries with preset distribution assumptions,
which can be used for further analysis.

```
> python -m archeo simulate-agnostic-black-hole-population --help

  Simulate a population of agnostic black hole binaries. The function
  simulates both aligned and precession spin configurations based on the
  user's choice.

  Command example: >> python -m archeo simulate-agnostic-black-hole-population
  --aligned-spin

Options:
  -n, --size INTEGER        Number of black holes to simulate.
  -np, --n-workers INTEGER  Number of cores to use for simulation.
  -o, --output-dir TEXT     Directory to save the generated data.
  -as, --aligned-spin       Toggle to simulate aligned spin binaries.
  --help                    Show this message and exit.
```

```
> python -m archeo simulate-second-generation-black-hole-population --help

Usage: python -m archeo simulate-second-generation-black-hole-population
           [OPTIONS]

  Simulate a population of second generation black hole binaries. The function
  simulates both aligned and precession spin configurations based on the
  user's choice.

  Command example: >> python -m archeo simulate-second-generation-black-hole-
  population --aligned-spin

Options:
  -n, --size INTEGER        Number of black holes to simulate.
  -np, --n-workers INTEGER  Number of cores to use for simulation.
  -o, --output-dir TEXT     Directory to save the generated data.
  -as, --aligned-spin       Toggle to simulate aligned spin binaries.
  --help                    Show this message and exit.
```

Here is an example of how to generate a preset prior using the CLI:

```bash
python -m archeo simulate-agnostic-black-hole-population --aligned-spin
```

After the commend is executed, we should see two files generated in the current directory:
- `simulated_binaries.parquet`: a parquet file containing the simulated binary properties (masses, spins, kick velocities, etc.).
- `binary_generator_config.json`: a json file containing the configuration used for the simulation.

### Visualize the Prior Distribution with CLI

We provide a command to visualize the generated black hole binary population properties,
which can be used to understand the prior distribution of black hole properties.

```
> python -m archeo visualize-black-hole-population --help

Usage: python -m archeo visualize-black-hole-population [OPTIONS]

  Generate some visualizations for a black hole population.

  Command example: >> python -m archeo visualize-black-hole-population
  --filepath ./simulated_binaries.parquet

Options:
  -f, --filepath TEXT    Path to the binary data.  [required]
  -o, --output-dir TEXT  Directory to save visualizations.
  --help                 Show this message and exit.
```

Here is an example of how to visualize the prior distribution using the CLI:

```bash
python -m archeo visualize-black-hole-population --filepath ./simulated_binaries.parquet
```

This command will read the binary data from `simulated_binaries.parquet` and save the visualization output in the current directory.
Note that the visualizations include:
- Animation of how distributions (various parameters) change over kick magnitude constraint.
- 2D histogram of the mass-spin distribution.
- Kick distribution for each spin-bin (binwidth=0.1).

## Ancestral Parameter Estimation

The following example demonstrates how to use the package to visualize the prior and posterior distributions of a single event.

```python
import archeo

# Load the mass/spin samples from a file (usually PE results from LVK)
# They are expected to be a list of floats
mass_posterior = [68.0, 71.4, ..., 91.4]
spin_posterior = [0.31, 0.54, ..., 0.64]

# Create a prior
pipeline = archeo.get_binary_generation_pipeline("2g_aligned_spin")
df_prior, binary_generator = pipeline(size=10000, n_workers=-1)
# Infer the posterior distribution
df_posterior = infer_ancestral_posterior_distribution(
    df_binaries=df_prior,
    mass_posterior_samples=mass_posterior,
    spin_posterior_samples=spin_posterior,
    n_workers=-1,
)

# Visualize the prior and the posterior
archeo.visualize_prior_distribution(df_prior, output_dir="./")
archeo.visualize_posterior_estimation({"GW190521": df_posterior}, output_dir="./")
```

## Documentation for Developers

To import archeo in your Python code, please refer to the documentation page at [https://wyhwong.github.io/archeo/](https://wyhwong.github.io/archeo/).

## Getting Help

The code is maintained by [Henry Wong](https://github.com/wyhwong) under [Juan Calderon Bustillo](https://git.ligo.org/juan.calderonbustillo)'s supervision. You can find the [list of contributors](https://github.com/wyhwong/archeo/graphs/contributors) here. Please report bugs by raising an issue on our [GitHub](https://github.com/wyhwong/archeo) repository.

## License

Archeo has a MIT License - see the [LICENSE](https://github.com/wyhwong/archeo/blob/main/LICENSE).
