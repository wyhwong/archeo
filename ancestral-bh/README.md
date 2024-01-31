# Ancestral BH

## LIGO JupyterLab Environment

```bash
# Clone the repository
git clone https://github.com/wyhwong/Ancestral-BH.git
cd ancestral-bh

# Setup for the environment
conda create -n ancestral-bh python=3.11
conda activate ancestral-bh
pip3 install poetry
make install

# Modify main.yml and prior.yml according to your needs
make run
```

---

## GNU Make Commands for Development

```bash
# Install dependencies in Poetry
make install

# Run simulation
make run

# Run static code analysis
# Components included:
#   - black (formatter)
#   - bandit (security linter)
#   - pylint (linter)
#   - mypy (type checker)
#   - isort (import sorter)
make analyze

# Update dependencies in Poetry
make update

# After developement
make format
```

---

## After Simulation

By default, the simulation will output the following:

1. Prior distribution of each parameter in feather format.
2. Posterior distribution of each parameter in feather format.
3. Data visualization of the prior distribution in PNG format.
4. Data visualization of the posterior distribution in PNG format.
