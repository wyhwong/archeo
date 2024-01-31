export DOCKER_BUILDKIT=1

# Environment variables
export LOGLEVEL?=10
export LOGFILE_PATH?=./runtime.log

export RESULTS_DIR?=./results
export MAX_MULTITHREAD_WORKER?=20
export CONFIG_PATH?=./main.yml
export PRIOR_CONFIG_PATH?=./prior.yml
export RANDOM_SEED?=2023

# Install Python dependencies in poetry environment
local:
	poetry install
	poetry run pre-commit install -f

# Run code in poetry environment
run:
	poetry run python3 src/main.py

# Upate Python dependencies in poetry environment
update:
	poetry update

# Run code analysis
analyze:
	poetry install
	poetry run pre-commit install -f
	poetry run black src --check
	poetry run bandit -r src -c pyproject.toml
	poetry run pylint src --rcfile ../.pylintrc
	poetry run isort src --check-only
	poetry run mypy src --show-error-codes
	poetry run radon cc src

# Format src using black, isort
format:
	poetry run black src
	poetry run isort src
