# PAPER
PArental Parameter Estimation of black-hole merger Remnant (PAPER) is a repository for parameter estimation on the parental masses, natal kick, and spin of the black holes in Pair-instability Supernova (PISN) gap.

---

# Runtime

### LIGO JupyterLab

```bash
# Setup for the environment
pip3 install pyyaml p_tqdm scipy seaborn numpy pandas matplotlib jupyterthemes notebook tables corner surfinbh

# Modify configs/main_config.yml and configs/prior_config.yml according to your needs
python3 main.py
```

---

### Docker environment

```bash
# Build Docker image
make build

# Start container (simulation)
make run

# For visualization or development
make jupyter_up
```

---

# Application

We can use the scripts in the repository to estimate the parental mass and kick of GW190521. Please find the related data of GW190521 in [LIGO Document P2000158-v4](https://dcc.ligo.org/LIGO-P2000158/public). After downloading the data and modify [main_config.yml](./configs/main_config.yml) accordingly, we can compute the results.

### Demonstration

### Corner plot of the first component black hole in GW190521

![plot](./images/GW190521_LVC_BH1_corner.png)


### Corner plot of the second component black hole in GW190521

![plot](./images/GW190521_LVC_BH2_corner.png)

# Authors
[@wyhwong](https://github.com/wyhwong), [@juan.calderonbustillo](https://git.ligo.org/juan.calderonbustillo)
