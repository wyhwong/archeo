# PAPER
PArental Parameter Estimation of black-hole merger Remnant (PAPER) is a repository for parameter estimation on the parental masses, natal kick, and spin of the black holes in Pair-instability Supernova (PISN) gap.

---

# Runtime

```bash
# For running in LIGO jupyter lab
# Setup for the environment
pip3 install pyyaml p_tqdm scipy seaborn numpy pandas matplotlib jupyterthemes notebook tables corner surfinbh

# Modify config/config.yml according to your needs
python3 main.py
```

---

```bash
# For Docker environment:
# Build Docker image
make build

# Start container
make run

# For visualization or development
make jupyter_up
```
