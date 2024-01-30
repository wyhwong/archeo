import os

# For logger
LOGLEVEL = int(os.getenv("LOGLEVEL", "10"))
LOGFILE_PATH = os.getenv("LOGFILE_PATH", "./runtime.log")

# For facade
RESULTS_DIR = os.getenv("RESULTS_DIR", "../results")
MAX_MULTIPROCESS_WORKER = int(os.getenv("MAX_MULTIPROCESS_WORKER", "20"))
CONFIG_PATH = os.getenv("CONFIG_PATH", "../setting.yml")
PRIOR_CONFIG_PATH = os.getenv("PRIOR_CONFIG_PATH", "../prior.yml")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "2023"))
