## ARCHEO - Inferring the natal kick and parental masses posterior of black holes in Pair-instability Supernova (PISN) gap.

[![github](https://img.shields.io/badge/GitHub-archeo-blue.svg)](https://github.com/wyhwong/archeo)
[![PyPI version](https://badge.fury.io/py/archeo.svg)](https://pypi.org/project/archeo/)
[![DOI](https://zenodo.org/badge/626377469.svg)](https://doi.org/10.5281/zenodo.14306853)
[![Downloads](https://img.shields.io/pepy/dt/archeo)](https://github.com/wyhwong/archeo)
[![Python version](https://img.shields.io/pypi/pyversions/archeo)](https://pypi.org/project/archeo/)
[![license](https://img.shields.io/badge/license-MIT-orange.svg)](https://github.com/wyhwong/archeo/blob/main/LICENSE)
[![CI](https://github.com/wyhwong/archeo/actions/workflows/main.yml/badge.svg)](https://github.com/wyhwong/archeo/actions/workflows/main.yml/)

## Basic Usage

The following example demonstrates how to use the package to visualize the prior and posterior distributions of a single event.

```python
import archeo

# Load the mass/spin samples from a file
# They are expected to be a list of floats
mass_posterior = [68.0, 71.4, ..., 91.4]
spin_posterior = [0.31, 0.54, ..., 0.64]

# Create a prior (preset priors are "precessing", "aligned_spin", and "positively_aligned_spin")
prior = archeo.Prior.from_config("precessing")
# Create a posterior from the samples and the prior
posterior = prior.to_posterior(mass_posterior, spin_posterior)

# Visualize the prior and the posterior
archeo.visualize_prior_distribution(prior, output_dir="./")
archeo.visualize_posterior_estimation({"GW190521": posterior}, output_dir="./")
```

## Configure your own prior

Check out the preset priors in [precessing.py](https://github.com/wyhwong/archeo/blob/main/src/archeo/preset/precessing.py) and [aligned_spin.py](https://github.com/wyhwong/archeo/blob/main/src/archeo/preset/aligned_spin.py). From that, one should be able to create their own prior by following the same structure.

# Credits
The code is maintained by [Henry Wong](https://github.com/wyhwong) under [Juan Calderon Bustillo](https://git.ligo.org/juan.calderonbustillo)'s supervision. You can find the [list of contributors](https://github.com/wyhwong/archeo/graphs/contributors) here. Please report bugs by raising an issue on our [GitHub](https://github.com/wyhwong/archeo) repository.
