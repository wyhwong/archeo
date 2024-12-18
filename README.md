# ARCHEO
[![PyPI version](https://badge.fury.io/py/archeo.svg)](https://pypi.org/project/archeo/)
[![DOI](https://zenodo.org/badge/626377469.svg)](https://doi.org/10.5281/zenodo.14306853)
[![Downloads](https://img.shields.io/pepy/dt/archeo)](https://github.com/wyhwong/archeo)
[![Python version](https://img.shields.io/pypi/pyversions/archeo)](https://pypi.org/project/archeo/)
[![license](https://img.shields.io/badge/license-MIT-orange.svg)](https://github.com/wyhwong/archeo/blob/main/LICENSE)
[![CI](https://github.com/wyhwong/archeo/actions/workflows/main.yml/badge.svg)](https://github.com/wyhwong/archeo/actions/workflows/main.yml/)


Archeo is a package for inferring the natal kick and parental masses posterior of black holes in Pair-instability Supernova (PISN) gap. We study the parental black holes of GW190521 and investigate the probability that the component black holes in GW190521 are a result of black hole merger. The methodology is described in [Methodology](#methodology) section, and it can be applied to any other gravitational wave event.

---

# Installation

The installation can be done via PyPI or from source.

## PyPI
```bash
pip3 install archeo
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

[1] Carlos Araújo Álvarez, Henry W. Y. Wong, Juan Calderón Bustillo. ["Kicking Time Back in Black Hole Mergers: Ancestral Masses, Spins, Birth Recoils, and Hierarchical-formation Viability of GW190521." The Astrophysical Journal 977.2 (2024): 220.](https://iopscience.iop.org/article/10.3847/1538-4357/ad90a9)

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

# First Application - GW190521

We can leverage the pacakge to estimate the parental mass and kick of component black holes in GW190521. Please find the related data of GW190521 in [LIGO Document P2000158-v4](https://dcc.ligo.org/LIGO-P2000158/public).

---

# Credits
The code is maintained by [Henry Wong](https://github.com/wyhwong) under [Juan Calderon Bustillo](https://git.ligo.org/juan.calderonbustillo)'s supervision. You can find the [list of contributors](https://github.com/wyhwong/archeo/graphs/contributors) here. Please report bugs by raising an issue on our [GitHub](https://github.com/wyhwong/archeo) repository.
