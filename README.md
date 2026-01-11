# ARCHEO

[![PyPI version](https://badge.fury.io/py/archeo.svg)](https://pypi.org/project/archeo/)
[![documentation](https://img.shields.io/badge/docs-archeo-blue.svg)](https://wyhwong.github.io/archeo/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17632389.svg)](https://doi.org/10.5281/zenodo.14306853)
[![Downloads](https://img.shields.io/pepy/dt/archeo)](https://github.com/wyhwong/archeo)
[![Python version](https://img.shields.io/pypi/pyversions/archeo)](https://pypi.org/project/archeo/)
[![license](https://img.shields.io/badge/license-MIT-orange.svg)](https://github.com/wyhwong/archeo/blob/main/LICENSE)
[![CI](https://github.com/wyhwong/archeo/actions/workflows/main.yml/badge.svg)](https://github.com/wyhwong/archeo/actions/workflows/main.yml/)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://archeo.streamlit.app/)

Archeo is a Python package designed to infer the natal kick, ancestral masses, and spins of black holes in the Pair-instability Supernova (PISN) gap,
with a particular focus on hierarchical black hole formation.

Our method applies to any binary black hole event detected via gravitational waves, enabling researchers to:
- Infer the parental (ancestral) black holes of observed binaries.
- Estimate the birth recoil velocities to determine if a black hole remains in its host environment or is ejected.
- Evaluate hierarchical merger scenarios to assess whether a black hole could be a product of previous mergers.

See more details at [https://wyhwong.github.io/archeo/methodology/](https://wyhwong.github.io/archeo/methodology/).

---

# Installation

Install via PyPI or from source.

## PyPI

```bash
# Basic installation (without UI)
pip3 install archeo
# If you want to use the web UI features (powered by Streamlit)
pip3 install archeo[ui]
```

## From source

```bash
git clone https://github.com/wyhwong/archeo.git
cd archeo/src

# If you use poetry
poetry install
# If you do not use poetry
pip3 install -r requirements.txt .
```

---

## Usage

For CLI usage, please refer to [README.md](./src/README.md) inside `src` folder.

To import archeo in your Python code, please refer to the documentation page at [https://wyhwong.github.io/archeo/](https://wyhwong.github.io/archeo/).

---

## Publications

Here we list the publications that have used Archeo:

[1] Carlos Araújo Álvarez, Henry W. Y. Wong, Juan Calderón Bustillo. "Kicking Time Back in Black Hole Mergers: Ancestral Masses, Spins, Birth Recoils, and Hierarchical-formation Viability of GW190521." [The Astrophysical Journal 977.2 (2024): 220.](https://iopscience.iop.org/article/10.3847/1538-4357/ad90a9)

[2] The LIGO Scientific Collaboration, the Virgo Collaboration, the KAGRA Collaboration. "GW231123: a Binary Black Hole Merger with Total Mass 190-265 $M_\odot$." [The Astrophysical Journal Letters 993 L25.](https://iopscience.iop.org/article/10.3847/2041-8213/ae0c9c)

[3] The LIGO Scientific Collaboration, the Virgo Collaboration, the KAGRA Collaboration. "GW241011 and GW241110: Exploring Binary Formation and Fundamental Physics with Asymmetric, High-spin Black Hole Coalescences." [The Astrophysical Journal Letters 993.1 (2025): L21.](https://iopscience.iop.org/article/10.3847/2041-8213/ae0d54)

---

## Getting Help

The code is maintained by [Henry Wong](https://github.com/wyhwong) under [Juan Calderon Bustillo](https://git.ligo.org/juan.calderonbustillo)'s supervision. You can find the [list of contributors](https://github.com/wyhwong/archeo/graphs/contributors) here. Please report bugs by raising an issue on our [GitHub](https://github.com/wyhwong/archeo) repository.
