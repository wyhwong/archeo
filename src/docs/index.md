# Archeo (V2)

[![PyPI version](https://badge.fury.io/py/archeo.svg)](https://pypi.org/project/archeo/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17632389.svg)](https://doi.org/10.5281/zenodo.14306853)
[![Downloads](https://img.shields.io/pepy/dt/archeo)](https://github.com/wyhwong/archeo)
[![Python version](https://img.shields.io/pypi/pyversions/archeo)](https://pypi.org/project/archeo/)
[![license](https://img.shields.io/badge/license-MIT-orange.svg)](https://github.com/wyhwong/archeo/blob/main/LICENSE)
[![CI](https://github.com/wyhwong/archeo/actions/workflows/ci.yml/badge.svg)](https://github.com/wyhwong/archeo/actions/workflows/ci.yml/)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://archeo.streamlit.app/)

Archeo is a Python package designed to infer the natal kick, ancestral masses, and spins of black holes in the Pair-instability Supernova (PISN) gap,
with a particular focus on hierarchical black hole formation.
Please check out our [methodology](https://wyhwong.github.io/archeo/methodology/) if you are interested.

In v2, we have reimplemented everything from scratch with a more modular and extensible design,
which allows users to easily customize the prior assumptions and the simulation configuration.
The most important note is that we have boosted the sampling speed dramatically.
**With a 8-core CPU (AMD Ryzen 7 9700X), we can now generate 2M samples of aligned spin binaries within 10 minutes.**
This is 10+x faster than in v1, where we needed hours to generate the same amount of samples.

---

## Publications

Here we list the publications that have used Archeo:

[1] Carlos Araújo Álvarez, Henry W. Y. Wong, Juan Calderón Bustillo. "Kicking Time Back in Black Hole Mergers: Ancestral Masses, Spins, Birth Recoils, and Hierarchical-formation Viability of GW190521." [The Astrophysical Journal 977.2 (2024): 220.](https://iopscience.iop.org/article/10.3847/1538-4357/ad90a9)

[2] The LIGO Scientific Collaboration, the Virgo Collaboration, the KAGRA Collaboration. "GW231123: a Binary Black Hole Merger with Total Mass 190-265 $M_\odot$." [The Astrophysical Journal Letters 993 L25.](https://iopscience.iop.org/article/10.3847/2041-8213/ae0c9c)

[3] The LIGO Scientific Collaboration, the Virgo Collaboration, the KAGRA Collaboration. "GW241011 and GW241110: Exploring Binary Formation and Fundamental Physics with Asymmetric, High-spin Black Hole Coalescences." [The Astrophysical Journal Letters 993.1 (2025): L21.](https://iopscience.iop.org/article/10.3847/2041-8213/ae0d54)

---

## Getting Help

The code is maintained by [Henry Wong](https://github.com/wyhwong) under [Juan Calderon Bustillo](https://git.ligo.org/juan.calderonbustillo)'s supervision. You can find the [list of contributors](https://github.com/wyhwong/archeo/graphs/contributors) here. Please report bugs by raising an issue on our [GitHub](https://github.com/wyhwong/archeo) repository.
