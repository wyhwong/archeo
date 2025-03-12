# ARCHEO
[![github](https://img.shields.io/badge/GitHub-archeo-blue.svg)](https://github.com/wyhwong/archeo)
[![PyPI version](https://badge.fury.io/py/archeo.svg)](https://pypi.org/project/archeo/)
[![DOI](https://zenodo.org/badge/626377469.svg)](https://doi.org/10.5281/zenodo.14306853)
[![Downloads](https://img.shields.io/pepy/dt/archeo)](https://github.com/wyhwong/archeo)
[![Python version](https://img.shields.io/pypi/pyversions/archeo)](https://pypi.org/project/archeo/)
[![license](https://img.shields.io/badge/license-MIT-orange.svg)](https://github.com/wyhwong/archeo/blob/main/LICENSE)
[![CI](https://github.com/wyhwong/archeo/actions/workflows/main.yml/badge.svg)](https://github.com/wyhwong/archeo/actions/workflows/main.yml/)

Archeo is a Python package designed to infer the natal kick, ancestral masses, and spins of black holes in the Pair-instability Supernova (PISN) gap,
with a particular focus on hierarchical black hole formation.

## Basic Usage

The following example demonstrates how to use the package to visualize the prior and posterior distributions of a single event.

```python
import archeo

# Load the mass/spin samples from a file (usually PE results from LVK)
# They are expected to be a list of floats
mass_posterior = [68.0, 71.4, ..., 91.4]
spin_posterior = [0.31, 0.54, ..., 0.64]

# Create a prior
prior = archeo.Prior.from_config("precessing_spin")
# Create a posterior from the samples and the prior
posterior = prior.to_posterior(mass_posterior, spin_posterior)

# Visualize the prior and the posterior
archeo.visualize_prior_distribution(prior, output_dir="./")
archeo.visualize_posterior_estimation({"GW190521": posterior}, output_dir="./")
```

## Available Preset Priors

This table provides an overview of the different prior configurations available in archeo.

| Name | Samples  | Fits Model | Spin Aligned | Only Up-Aligned Spin | $\chi_1$ | $\chi_2$ | $\phi_1$ [rad] | $\phi_2$ [rad] | $\theta_1$ [rad] | $\theta_2$ [rad] | $m_1 [M_\odot]$ | $m_2 [M_\odot]$ | $q$ | Uniform in $q$ |
|------------------------------------|-----------|------------------|----|-----|-------|-------|------------|------------|-----------|-----------|---------|---------|-------|-----|
| agnostic_precessing_spin (default) | 2,000,000 | NRSur7dq4Remnant | ❌ | ❌ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | 5 - 200 | 5 - 200 | 1 - 6 | ❌ |
| agnostic_aligned_spin              | 2,000,000 | NRSur3dq8Remnant | ✅ | ❌ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | 5 - 200 | 5 - 200 | 1 - 6 | ❌ |
| precessing_spin                    | 2,000,000 | NRSur7dq4Remnant | ❌ | ❌ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | 5 - 65  | 5 - 65  | 1 - 6 | ❌ |
| aligned_spin                       | 2,000,000 | NRSur3dq8Remnant | ✅ | ❌ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | 5 - 65  | 5 - 65  | 1 - 6 | ❌ |
| positively_aligned_spin            | 2,000,000 | NRSur3dq8Remnant | ✅ | ✅ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | 5 - 65  | 5 - 65  | 1 - 6 | ❌ |

## Configure your own prior

Check out the preset priors in [precessing.py](https://github.com/wyhwong/archeo/blob/main/src/archeo/preset/precessing.py) and [aligned.py](https://github.com/wyhwong/archeo/blob/main/src/archeo/preset/aligned.py). From that, one should be able to create their own prior by following the same structure.

## Try our UI

Archeo also provides a simple web-based user interface to visualize the distributions of remnant properties.
To run the UI locally, simply run the following command:

```bash
pip3 install archeo[ui]
python3 -m archeo.ui
```

Then the UI will be available at [localhost:8501](http://localhost:8501).

## Getting Help

The code is maintained by [Henry Wong](https://github.com/wyhwong) under [Juan Calderon Bustillo](https://git.ligo.org/juan.calderonbustillo)'s supervision. You can find the [list of contributors](https://github.com/wyhwong/archeo/graphs/contributors) here. Please report bugs by raising an issue on our [GitHub](https://github.com/wyhwong/archeo) repository.

## License

Archeo has a MIT License - see the [LICENSE](https://github.com/wyhwong/archeo/blob/main/LICENSE).
