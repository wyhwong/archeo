#!/usr/bin/env python3
import numpy as np
import pandas as pd
from glob import glob

from utils.prior_simulation import simulate_binaries
from utils.parental_params_estimation import get_parental_params_likelihood
from utils.statistical_operation import convert_posterior_to_likelihood
from utils.common import (
    read_posterior_from_h5,
    read_posterior_from_json,
    get_main_config,
    check_and_create_dir,
    save_dict_as_yml,
)
from utils.visualization import plot_parameter_estimation, plot_posterior_corner
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
    if CONFIG["posterior"]["posteriorJson"]:
        for label, posterior_filepath in CONFIG["posterior"]["posteriorJson"].items():
            posteriors[label] = read_posterior_from_json(filepath=posterior_filepath)
    if CONFIG["posterior"]["posteriorH5"]:
        for label, posterior_filepath in CONFIG["posterior"]["posteriorH5"].items():
            posteriors[label] = read_posterior_from_h5(filepath=posterior_filepath)

    # Parameter Estimation
    for label, posterior in posteriors.items():
        for bh_component_index in range(1, 3):
            if CONFIG["estimation"]["enable"]:
                posterior_label = f"{label},BH{bh_component_index}"
                parental_mass_likelihoods = get_parental_params_likelihood(
                    prior_df=prior_df,
                    child_spin_posterior=posterior[f"a_{bh_component_index}"],
                    child_mass_posterior=posterior[f"mass_{bh_component_index}_source"],
                    posterior_label=posterior_label,
                    output_dir=OUTPUTDIR,
                    sampling_spin_binwidth=CONFIG["estimation"]["samplingSpinBinwidth"],
                    sample_size=CONFIG["estimation"]["sampleSize"],
                    nbins=CONFIG["estimation"]["nbins"],
                )
                child_mass_likelihood = convert_posterior_to_likelihood(
                    posterior=posterior[f"mass_{bh_component_index}_source"],
                    weights=None,
                    posterior_label=posterior_label,
                    nbins=CONFIG["estimation"]["nbins"],
                )
                likelihoods = [child_mass_likelihood, parental_mass_likelihoods["p1"], parental_mass_likelihoods["p2"]]
                plot_parameter_estimation(
                    prior_df=None,
                    target_parameter="parental_mass",
                    target_parameter_label="Mass $(M_{\odot})$",
                    likelihoods=likelihoods,
                    plot_label=posterior_label,
                    output_dir=OUTPUTDIR,
                )

            if CONFIG["estimation"]["plotCorner"]:
                posterior_df = pd.read_hdf(f"{OUTPUTDIR}/{label},BH{bh_component_index}_parental_mass_estimates.h5")
                plot_posterior_corner(
                    posterior_df=posterior_df,
                    posterior_label=posterior_label,
                    var_names=["vf", "m1", "m2", "chif"],
                    labels=["$v_f$", "$m_1$", "$m_2$", "$\chi_f$"],
                    nbins=CONFIG["estimation"]["nbins"],
                    output_dir=OUTPUTDIR,
                )


if __name__ == "__main__":
    main()
