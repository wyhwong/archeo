export DOCKER_BUILDKIT=1

# For Docker image
export VERSION?=devel
export USERNAME?=$(shell whoami)
export USER_ID?=$(shell id -u)
export GROUP_ID?=$(shell id -g)

# Environment variables
export TZ?=Asia/Hong_Kong
export LOGLEVEL?=10

# For Code analysis
export TEST_REPORT_DIR?=./test_report

# Build Docker image
build:
	docker build -t TBC:${VERSION} \
	--build-arg USERNAME=${USERNAME} \
	--build-arg USER_ID=${USER_ID} \
	--build-arg GROUP_ID=${GROUP_ID} \
	--build-arg TZ=${TZ} \
	--no-cache .

# Run Docker container
run:
	docker run --rm -it -d --name TBC \
	--env LOGLEVEL=${LOGLEVEL} \
	--env TZ=${TZ} \
	-p ${PORT}:8888 \
	TBC:${VERSION}

# Install Python dependencies in poetry environment
local:
	poetry install
	poetry run pre-commit install -f

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
	poetry run pytest --disable-warnings --cov-report=html:${TEST_REPORT_DIR} --cov=src/core src
	open ${TEST_REPORT_DIR}/index.html

# Upate Python dependencies in poetry environment
update:
	poetry update

# Format src using black, isort
format:
	poetry run black src
	poetry run isort src

# Run test cases in poetry environment and open coverage report
test:
	poetry run pytest --disable-warnings --cov-report=html:${TEST_REPORT_DIR} --cov=src/core src
	open ${TEST_REPORT_DIR}/index.html
