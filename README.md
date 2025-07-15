# ARCHEO
[![PyPI version](https://badge.fury.io/py/archeo.svg)](https://pypi.org/project/archeo/)
[![DOI](https://zenodo.org/badge/626377469.svg)](https://doi.org/10.5281/zenodo.14306853)
[![Downloads](https://img.shields.io/pepy/dt/archeo)](https://github.com/wyhwong/archeo)
[![Python version](https://img.shields.io/pypi/pyversions/archeo)](https://pypi.org/project/archeo/)
[![license](https://img.shields.io/badge/license-MIT-orange.svg)](https://github.com/wyhwong/archeo/blob/main/LICENSE)
[![CI](https://github.com/wyhwong/archeo/actions/workflows/main.yml/badge.svg)](https://github.com/wyhwong/archeo/actions/workflows/main.yml/)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://archeo.streamlit.app/)

Archeo is a Python package designed to infer the natal kick, ancestral masses, and spins of black holes in the Pair-instability Supernova (PISN) gap,
with a particular focus on hierarchical black hole formation.

Our method (described in [Methodology](#methodology)) applies to any binary black hole event detected via gravitational waves, enabling researchers to:
- Infer the parental (ancestral) black holes of observed binaries.
- Estimate the birth recoil velocities to determine if a black hole remains in its host environment or is ejected.
- Evaluate hierarchical merger scenarios to assess whether a black hole could be a product of previous mergers.

---

# Installation

Install via PyPI or from source.

## PyPI
```bash
# Basic installation (without UI)
pip3 install archeo
# If you want to use the UI
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
Please see [README.md](./src/README.md) inside `src` folder.

---

## Methodology

The methodology is describe in the following paper:

[1] Carlos Araújo Álvarez, Henry W. Y. Wong, Juan Calderón Bustillo. "Kicking Time Back in Black Hole Mergers: Ancestral Masses, Spins, Birth Recoils, and Hierarchical-formation Viability of GW190521." [The Astrophysical Journal 977.2 (2024): 220.](https://iopscience.iop.org/article/10.3847/1538-4357/ad90a9)

#### Bibtex Citation:
```bibtex
@article{araujo2024kicking,
  title={Kicking Time Back in Black Hole Mergers: Ancestral Masses, Spins, Birth Recoils, and Hierarchical-formation Viability of GW190521},
  author={Ara{\'u}jo-{\'A}lvarez, Carlos and Wong, Henry WY and Liu, Anna and Bustillo, Juan Calder{\'o}n},
  journal={The Astrophysical Journal},
  volume={977},
  number={2},
  pages={220},
  year={2024},
  publisher={IOP Publishing}
}
```

---

## Publications

Here we list the publications that have used Archeo:

[1] Carlos Araújo Álvarez, Henry W. Y. Wong, Juan Calderón Bustillo. "Kicking Time Back in Black Hole Mergers: Ancestral Masses, Spins, Birth Recoils, and Hierarchical-formation Viability of GW190521." [The Astrophysical Journal 977.2 (2024): 220.](https://iopscience.iop.org/article/10.3847/1538-4357/ad90a9)

[2] The LIGO Scientific Collaboration, the Virgo Collaboration, the KAGRA Collaboration. "GW231123: a Binary Black Hole Merger with Total Mass 190-265 $M_\odot$." [arXiv preprint arXiv:2507.08219.](https://arxiv.org/abs/2507.08219)

---

## Getting Help

The code is maintained by [Henry Wong](https://github.com/wyhwong) under [Juan Calderon Bustillo](https://git.ligo.org/juan.calderonbustillo)'s supervision. You can find the [list of contributors](https://github.com/wyhwong/archeo/graphs/contributors) here. Please report bugs by raising an issue on our [GitHub](https://github.com/wyhwong/archeo) repository.
