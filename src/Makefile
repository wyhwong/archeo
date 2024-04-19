export DOCKER_BUILDKIT=1

# Environment variables
export LOGLEVEL?=20
export LOGFILE_PATH?=./runtime.log

export RESULTS_DIR?=./results
export MAX_MULTITHREAD_WORKER?=20
export CONFIG_PATH?=./main.yml
export PRIOR_CONFIG_PATH?=./prior.yml
export RANDOM_SEED?=2023

# Install Python dependencies in poetry environment
install:
	poetry install
	poetry run pre-commit install -f

# Run code in poetry environment
run:
	poetry run python3 ancestral-bh/main.py

# Upate Python dependencies in poetry environment
update:
	poetry update

# Run code analysis
analyze:
	poetry install
	poetry run pre-commit install -f
	poetry run black archeo --check
	poetry run bandit -r archeo -c pyproject.toml
	poetry run pylint archeo --rcfile ../.pylintrc
	poetry run isort archeo --check-only
	poetry run mypy archeo --show-error-codes
	poetry run radon cc archeo

# Format archeo using black, isort
format:
	poetry run black archeo
	poetry run isort archeo
