import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait

from .logger import get_logger
from .common import get_prior_config, save_dict_as_yml
from .binaries import BinaryParamsGenerator, load_fits, Binary
from .visualization import (
    plot_prior_params_distribution,
    plot_prior_kick_against_spin,
    plot_prior_kick_distribution_on_spin,
)

LOGGER = get_logger(logger_name="Utils | Prior Simulation")


def simulate_binary(binary_marker: str, binary_params: tuple) -> list:
    massRatio, chi1, chi2, mass1, mass2 = binary_params
    LOGGER.debug(f"Simulating binary {binary_marker}, with {massRatio=}, {chi1=}, {chi2=}")
    binary = Binary(fits=FITS, massRatio=massRatio, chi1=chi1, chi2=chi2)
    remnant_params = binary.merge()
    LOGGER.debug(f"Simulated binary {binary_marker}, {remnant_params=}")
    return [mass1, mass2] + remnant_params


# This function simulates a certain number of binaries according to the prior config in config/prior_config.yml.
# It outputs the prior as a .csv file in the output user defined directory.
def simulate_binaries(output_dir: str) -> list:
    global FITS
    LOGGER.info("Getting prior by simulation...")

    # Get prior config and save in output directory
    prior_config = get_prior_config()
    save_dict_as_yml(path=f"{output_dir}/prior_config.yml", input_dict=prior_config)
    merger_config = prior_config["merger"]
    binary_generator = BinaryParamsGenerator(config=merger_config)
    num_binaries = prior_config["amount"]
    FITS = load_fits(fits_name=prior_config["fits"])
    futures, prior = [], []

    # Simulate the mergers with the generated binary configs using multi-processing
    with ProcessPoolExecutor(max_workers=20) as Executor:
        for binary_marker in range(num_binaries):
            binary_params = binary_generator()
            futures.append(Executor.submit(simulate_binary, binary_marker, binary_params))
    wait(futures)

    # Process the results to a dataframe
    for _, remnantParams in enumerate(futures):
        prior.append(remnantParams.result())
    prior = pd.DataFrame(
        prior, columns=["m1", "m2", "mr", "chi1", "chi2", "mf", "mfError", "chif", "chifError", "vf", "vfError"]
    )

    # Some descriptive visualization of the prior
    remnant_params_dataframe = prior.drop(columns=["m1", "m2", "mr", "chi1", "chi2", "mfError", "chifError", "vfError"])
    remnant_params_dataframe.columns = ["$m_f$", "$\chi_f$", "$v_f$"]
    plot_prior_params_distribution(dataframe=remnant_params_dataframe, output_dir=output_dir, savefig=True)
    plot_prior_kick_against_spin(dataframe=remnant_params_dataframe, output_dir=output_dir, savefig=True)
    plot_prior_kick_distribution_on_spin(
        dataframe=remnant_params_dataframe, nbins=10, spin_max=1.0, spin_min=0.0, output_dir=output_dir, savefig=True
    )

    # Remove mass_1 and mass_2 if mass injection is not on
    if not prior_config["merger"]["massInjection"]:
        prior = prior.drop(columns=["m1", "m2"])

    # Save the prior as a csv file
    prior.to_csv(f"{output_dir}/prior.csv", index=False)
    LOGGER.info(f"Simulated prior saved at {output_dir}/prior.csv.")
    return prior
