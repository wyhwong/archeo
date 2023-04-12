#!/usr/bin/env python3
import numpy as np
import pandas as pd
from glob import glob

from utils.prior_simulation import simulate_binaries
from utils.common import get_main_config, check_and_create_dir, save_dict_as_yml
from utils.logger import get_logger


CONFIG = get_main_config()

# Setup of output directory
RESULTSDIR = "./results"
check_and_create_dir(dirpath=RESULTSDIR)
RUNLABEL = f"{int(len(glob(f'{RESULTSDIR}/*')) + 1)}_{CONFIG['runlabel']}"
OUTPUTDIR = f"{RESULTSDIR}/{RUNLABEL}"
check_and_create_dir(dirpath=OUTPUTDIR)

# Save current config
save_dict_as_yml(savepath=f"{OUTPUTDIR}/main_config.yml", input_dict=CONFIG)

# Set seed and logger
np.random.seed(seed=CONFIG["seed"])
LOGGER = get_logger(logger_name="Main", logfilePath=f"{OUTPUTDIR}/runtime.log")


def main() -> None:
    # Prior: simulation or read from csv
    if CONFIG["prior"]["loadResults"]:
        prior = pd.read_csv(CONFIG["prior"]["pathToCSV"])
        LOGGER.info("Loaded prior from previous experiment.")
    else:
        prior = simulate_binaries(output_dir=OUTPUTDIR)


if __name__ == "__main__":
    main()
