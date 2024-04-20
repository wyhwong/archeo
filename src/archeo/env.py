import os


# For logger
LOG_LEVEL = int(os.getenv("LOG_LEVEL", "20"))
LOG_FILEPATH = os.getenv("LOG_FILEPATH", "./runtime.log")
# NOTE: Although LOG_FMT and LOG_DATEFMT are in env.py, we do not expect
#       them to be changed by environment variables. They define the logging
#       style of archeo and should not be changed.
LOG_FMT = "%(asctime)s [%(name)s | %(levelname)s]: %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"

# For facade
RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")
MAX_MULTITHREAD_WORKER = int(os.getenv("MAX_MULTITHREAD_WORKER", "20"))
CONFIG_PATH = os.getenv("CONFIG_PATH", "./main.yml")
PRIOR_CONFIG_PATH = os.getenv("PRIOR_CONFIG_PATH", "./prior.yml")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "2023"))
