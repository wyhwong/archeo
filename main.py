#!/usr/bin/env python3
import numpy as np
import pandas as pd
from glob import glob

from utils.prior_simulation import simulate_binaries
from utils.kick_estimation import get_kick_likelihood
from utils.common import (
    read_posterior_from_h5,
    read_posterior_from_json,
    get_main_config,
    check_and_create_dir,
    save_dict_as_yml,
)
from utils.visualization import plot_parameter_estimation
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
LOGGER = get_logger(logger_name="Main", log_filepath=f"{OUTPUTDIR}/runtime.log")


def main() -> None:
    # Prior: simulation or read from csv
    if CONFIG["prior"]["loadResults"]:
        prior_df = pd.read_csv(CONFIG["prior"]["pathToCSV"])
        LOGGER.info("Loaded prior from previous experiment.")
    else:
        prior_df = simulate_binaries(output_dir=OUTPUTDIR)

    # Posterior: from json files or h5 files
    posteriors = {}
    for label, posterior_filepath in CONFIG["posterior"]["posteriorJson"].item():
        posteriors[label] = read_posterior_from_json(filepath=posterior_filepath)
    for label, posterior_filepath in CONFIG["posterior"]["posteriorH5"].item():
        posteriors[label] = read_posterior_from_h5(filepath=posterior_filepath)

    # Kick estimation
    kick_likelihoods = []
    for label, posterior in posteriors.item():
        for bh_component in range(1, 3):
            if CONFIG["estimation"]["kick"]["enable"]:
                kick_likelihood = get_kick_likelihood(
                    prior_df=prior_df,
                    spin_posterior=posterior[f"a_{bh_component}"],
                    posterior_label=f"{label},BH{bh_component}",
                    output_dir=OUTPUTDIR,
                )
                kick_likelihoods.append(kick_likelihood)
    plot_parameter_estimation(
        prior_df=prior_df,
        target_parameter="vf",
        target_parameter_label="$v_f$",
        likelihoods=kick_likelihoods,
        plot_label="all_in_one",
        output_dir=OUTPUTDIR,
        savefig=True,
    )


if __name__ == "__main__":
    main()
