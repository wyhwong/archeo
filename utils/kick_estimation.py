import pandas as pd
from time import time
from tqdm import tqdm

from .statistical_operation import compute_posterior_statistics
from .logger import get_logger

LOGGER = get_logger(logger_name="Utils | Kick Estimation")


def estimate_kick_by_spin(
    prior_df: pd.DataFrame,
    spin_posterior: list,
    nbins: int,
    savecsv=False,
    posterior_label=None,
    output_dir=None,
) -> pd.DataFrame:
    estimation_start_time = time()
    kick_posterior = pd.DataFrame(prior_df["vf"])
    kick_posterior["weights"] = 0

    spin_binwidth = (prior_df["chif"].max() - prior_df["chif"].min()) / nbins
    spin_min = prior_df["chif"].min()

    LOGGER.info("Recovering natal kick from spin measurements...")
    for spin_measurement in tqdm(spin_posterior):
        bin_index = round((spin_measurement - spin_min) / spin_binwidth)
        spin_min_in_bin = spin_min + bin_index * spin_binwidth
        spin_max_in_bin = spin_min + (bin_index + 1) * spin_binwidth
        sample_id_in_prior = prior_df.loc[
            (prior_df["chif"] >= spin_min_in_bin) & (prior_df["chif"] <= spin_max_in_bin)
        ].index
        if len(sample_id_in_prior) > 0:
            kick_posterior.loc[sample_id_in_prior, "weights"] += 1 / len(sample_id_in_prior)
        else:
            LOGGER.warning("No samples in prior bin, not enough of samples.")

    LOGGER.debug(f"Computational time for kick estimation: {(time() - estimation_start_time):.1f} seconds.")
    if savecsv:
        if posterior_label is None:
            raise ValueError("posterior_label must not be empty if savecsv is True.")
        if output_dir is None:
            raise ValueError("output_dir must not be None if savecsv is True.")
        filepath = f"{output_dir}/{posterior_label}_kick_estimation.csv"
        kick_posterior.to_csv(filepath, index=False)
        LOGGER.debug(f"Saved the estimated posterior to {filepath}.")
    return kick_posterior


def get_kick_likelihood(prior_df: pd.DataFrame, spin_posterior: list, posterior_label: str, output_dir: str, nbins=200):
    kick_likelihood = {}
    kick_posterior = estimate_kick_by_spin(
        prior_df=prior_df,
        spin_posterior=spin_posterior,
        nbins=nbins,
        savecsv=True,
        posterior_label=posterior_label,
        output_dir=output_dir,
    )
    (
        kick_likelihood["values"],
        kick_likelihood["edges"],
        error_lower_bound,
        error_upper_bound,
        kick_median,
    ) = compute_posterior_statistics(posterior=kick_posterior["vf"], weights=kick_posterior["weights"], nbins=nbins)
    kick_likelihood["label"] = "%s: $%d_{-%d}^{+%d}$ $km/s$" % (
        posterior_label,
        kick_median,
        error_lower_bound,
        error_upper_bound,
    )
    return kick_likelihood
