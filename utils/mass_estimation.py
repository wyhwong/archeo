import pandas as pd
import numpy as np
from time import time
from p_tqdm import p_map, t_imap

from .statistical_operation import compute_posterior_statistics
from .logger import get_logger

LOGGER = get_logger(logger_name="Utils | Mass Estimation")


def dummy_function(input):
    return input


# This function samples a certain number of samples from prior
# Then we compute the parental component mass of the samples
# The output is a tuple of:
#     1. list of mass of the heavier parental black hole (unit in solar mass)
#     2. list of mass of the lighter parental black hole (unit in solar mass)
#     3. list of kick of the child black hole (unit in km/s)
#     4. list of spin of the child black hole
def compute_parental_mass(
    prior_df: pd.DataFrame,
    child_mass_measurement: float,
    child_spin_measurement: float,
    prior_spin_binwidth: float,
    prior_spin_min: float,
    sample_size: int,
):
    bin_index = round((child_spin_measurement - prior_spin_min) / prior_spin_binwidth)
    spin_min_in_bin = prior_spin_min + bin_index * prior_spin_binwidth
    spin_max_in_bin = prior_spin_min + (bin_index + 1) * prior_spin_binwidth
    samples_in_prior = prior_df.loc[(prior_df["chif"] > spin_min_in_bin) & (prior_df["chif"] < spin_max_in_bin)]
    if len(samples_in_prior.index) > sample_size:
        sample_id_in_prior = samples_in_prior.sample(n=sample_size).index
    else:
        sample_id_in_prior = samples_in_prior.index
        LOGGER.warning(f"Not enough samples ({len(sample_id_in_prior)}) in prior bin, needed: {sample_size}")
    mf = child_mass_measurement / prior_df.loc[sample_id_in_prior, "mf"].values
    parental_mass_ratio = prior_df.loc[sample_id_in_prior, "mr"].values
    child_kick = prior_df.loc[sample_id_in_prior, "vf"].values
    child_spin = prior_df.loc[sample_id_in_prior, "chif"].values
    parental_mass_1 = mf * parental_mass_ratio / (parental_mass_ratio + 1)
    parental_mass_2 = mf / (parental_mass_ratio + 1)
    return (parental_mass_1, parental_mass_2, child_kick, child_spin)


def estimate_parental_mass_by_spin(
    prior_df: pd.DataFrame,
    child_spin_posterior: list,
    child_mass_posterior: list,
    nbins: int,
    sample_size: int,
    savehdf=False,
    posterior_label=None,
    output_dir=None,
) -> pd.DataFrame:
    estimation_start_time = time()
    prior_spin_binwidth = (prior_df["chif"].max() - prior_df["chif"].min()) / nbins
    prior_spin_min = prior_df["chif"].min()

    # Prepare dummy input arrays for multi-processing
    # We can also use global variables to do the same
    # Here we use a generator to spare the ram used to construct the array
    prior_df = t_imap(dummy_function, [prior_df] * len(child_spin_posterior))
    prior_spin_binwidth = t_imap(dummy_function, [prior_spin_binwidth] * len(child_spin_posterior))
    prior_spin_min = t_imap(dummy_function, [prior_spin_min] * len(child_spin_posterior))
    sample_size = t_imap(dummy_function, [sample_size] * len(child_spin_posterior))

    LOGGER.info("Recovering parental mass from spin measurements...")
    parmater_estimates = np.hstack(
        p_map(
            compute_parental_mass,
            prior_df,
            child_mass_posterior,
            child_spin_posterior,
            prior_spin_binwidth,
            prior_spin_min,
            sample_size,
        )
    )
    parental_mass_estimates_1, parental_mass_estimates_2, _, _ = parmater_estimates
    LOGGER.info(f"Computational time for the estimation: {time() - estimation_start_time:.1f} seconds.")
    if savehdf:
        if posterior_label is None:
            raise ValueError("posterior_label must not be empty if savecsv is True.")
        if output_dir is None:
            raise ValueError("output_dir must not be None if savecsv is True.")
        filepath = f"{output_dir}/{posterior_label}_parental_mass_estimates.h5"
        pd.DataFrame(parmater_estimates.T, columns=["m1", "m2", "vf", "chif"]).to_hdf(
            filepath, key="estimates", index=False
        )
        LOGGER.debug(f"Saved the estimated posterior to {filepath}.")
    return (parental_mass_estimates_1, parental_mass_estimates_2)


def get_parental_mass_likelihood():
    pass
