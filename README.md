# Ancestral BH
Ancestral BH is a repository for inferring the natal kick and parental masses posterior of black holes in Pair-instability Supernova (PISN) gap. We study the parental black holes of GW190521 and investigate the probability that the component black holes in GW190521 are a result of black hole merger.

---

## Prerequisites
- Poetry: [https://python-poetry.org](https://python-poetry.org)
- GNU make: [https://www.gnu.org/software/make/manual/make.html](https://www.gnu.org/software/make/manual/make.html)

---

## Usage
Please see [README.md](./ancestral-bh/README.md) inside `ancestral-bh` folder.

---

# Application - GW190521

We can use the scripts in the repository to estimate the parental mass and kick of component black holes in GW190521. Please find the related data of GW190521 in [LIGO Document P2000158-v4](https://dcc.ligo.org/LIGO-P2000158/public). After downloading the data and modify [main.yml](./ancestral-bh/main.yml) accordingly, we can compute the results. One can also tune the prior settings at [prior.yml](./ancestral-bh/prior.yml) accordingly.

---

# Authors
[@wyhwong](https://github.com/wyhwong), [@juan.calderonbustillo](https://git.ligo.org/juan.calderonbustillo)
