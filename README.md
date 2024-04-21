# ARCHEO
ARCHEO is a repository for inferring the natal kick and parental masses posterior of black holes in Pair-instability Supernova (PISN) gap. We study the parental black holes of GW190521 and investigate the probability that the component black holes in GW190521 are a result of black hole merger.

---

## Prerequisites
- Poetry: [https://python-poetry.org](https://python-poetry.org)
- GNU make: [https://www.gnu.org/software/make/manual/make.html](https://www.gnu.org/software/make/manual/make.html)

---

## Usage
Please see [README.md](./src/README.md) inside `src` folder.

---

## Methodology

The methodology is describe in the following paper:

[1] Carlos Araújo Álvarez, Henry W. Y. Wong, Juan Calderón Bustillo, [arxiv:2404.00720](https://arxiv.org/abs/2404.00720)

#### Bibtex Citation:
```bibtex
@article{alvarez2024kicking,
  title={Kicking time back in black-hole mergers: Ancestral masses, spins, birth recoils and hierarchical-formation viability of GW190521},
  author={{\'A}lvarez, Carlos Ara{\'u}jo and Wong, Henry WY and Bustillo, Juan Calder{\'o}n},
  journal={arXiv preprint arXiv:2404.00720},
  year={2024}
}
```

---

# Application - GW190521

We can use the scripts in the repository to estimate the parental mass and kick of component black holes in GW190521. Please find the related data of GW190521 in [LIGO Document P2000158-v4](https://dcc.ligo.org/LIGO-P2000158/public). After downloading the data and modify [main.yml](./src/main.yml) accordingly, we can compute the results. One can also tune the prior settings at [prior.yml](./src/prior.yml) accordingly.

---

# Authors
[@wyhwong](https://github.com/wyhwong), [@juan.calderonbustillo](https://git.ligo.org/juan.calderonbustillo)
