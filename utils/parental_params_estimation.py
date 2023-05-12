import pandas as pd
import numpy as np
from time import time
from p_tqdm import p_map, t_imap

from .statistical_operation import convert_posterior_to_likelihood
from .logger import get_logger

LOGGER = get_logger(logger_name="Utils | Parental Params Estimation")


def dummy_function(input):
    return input


# This function samples a certain number of samples from prior
# Then we compute the parental component mass of the samples
# The output is a tuple of:
#     1. list of mass of the heavier parental black hole (unit in solar mass)
#     2. list of mass of the lighter parental black hole (unit in solar mass)
#     3. list of kick of the child black hole (unit in km/s)
#     4. list of spin of the child black hole
def compute_parental_params(
    prior_df: pd.DataFrame,
    child_mass_measurement: float,
    child_spin_measurement: float,
    sampling_spin_binwidth: float,
    sample_size: int,
):
    spin_min_in_bin = child_spin_measurement - sampling_spin_binwidth
    spin_max_in_bin = child_spin_measurement + sampling_spin_binwidth
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


def estimate_parental_params_by_spin(
    prior_df: pd.DataFrame,
    child_spin_posterior: list,
    child_mass_posterior: list,
    sampling_spin_binwidth: float,
    sample_size: int,
    savehdf=False,
    posterior_label=None,
    output_dir=None,
) -> tuple:
    estimation_start_time = time()

    # Prepare dummy input arrays for multi-processing
    # We can also use global variables to do the same
    # Here we use a generator to spare the ram used to construct the array
    prior_df = t_imap(dummy_function, [prior_df] * len(child_spin_posterior))
    sampling_spin_binwidth = t_imap(dummy_function, [sampling_spin_binwidth] * len(child_spin_posterior))
    sample_size = t_imap(dummy_function, [sample_size] * len(child_spin_posterior))

    LOGGER.info("Recovering parental mass from spin measurements...")
    parmater_estimates = np.hstack(
        p_map(
            compute_parental_params,
            prior_df,
            child_mass_posterior,
            child_spin_posterior,
            sampling_spin_binwidth,
            sample_size,
        )
    )
    parmater_estimates = pd.DataFrame(parmater_estimates.T, columns=["m1", "m2", "vf", "chif"])
    LOGGER.info(f"Computational time for the estimation: {time() - estimation_start_time:.1f} seconds.")
    if savehdf:
        if posterior_label is None:
            raise ValueError("posterior_label must not be empty if savecsv is True.")
        if output_dir is None:
            raise ValueError("output_dir must not be None if savecsv is True.")
        filepath = f"{output_dir}/{posterior_label}_parental_mass_estimates.h5"
        parmater_estimates.to_hdf(filepath, key="estimates", index=False)
        LOGGER.debug(f"Saved the estimated posterior to {filepath}.")
    return parmater_estimates


def get_parental_params_likelihood(
    prior_df: pd.DataFrame,
    child_spin_posterior: list,
    child_mass_posterior: list,
    sampling_spin_binwidth: float,
    posterior_label: str,
    nbins: int,
    output_dir: str,
    sample_size=10,
):
    parental_params_likelihoods = {}
    parental_params_estimates = estimate_parental_params_by_spin(
        prior_df=prior_df,
        child_spin_posterior=child_spin_posterior,
        child_mass_posterior=child_mass_posterior,
        sampling_spin_binwidth=sampling_spin_binwidth,
        sample_size=sample_size,
        savehdf=True,
        posterior_label=posterior_label,
        output_dir=output_dir,
    )
    for parental_bh_index in range(1, 3):
        parental_params_likelihoods[f"p{parental_bh_index}"] = convert_posterior_to_likelihood(
            posterior=parental_params_estimates[f"m{parental_bh_index}"].values,
            posterior_label=f"Parental BH {parental_bh_index} (Child: {posterior_label})",
            weights=None,
            nbins=nbins,
            unit="$M_{\odot}$",
        )
    return parental_params_likelihoods
