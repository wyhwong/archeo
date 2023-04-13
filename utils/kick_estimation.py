import pandas as pd
from time import time
from tqdm import tqdm

from .logger import get_logger

LOGGER = get_logger(logger_name="Utils | Kick Estimation")


def estimate_kick_by_spin(
    prior: pd.DataFrame,
    spin_posterior: list,
    nbins: int,
    savecsv=False,
    posterior_label=None,
    output_dir=None,
) -> pd.DataFrame:
    estimation_start_time = time()
    kick_posterior = pd.DataFrame(prior["vf"])
    kick_posterior["weights"] = 0

    spin_binwidth = (prior["chif"].max() - prior["chif"].min()) / nbins
    spin_min = prior["chif"].min()
    for spin_measurement in tqdm(spin_posterior):
        bin_index = round((spin_measurement - spin_min) / spin_binwidth)
        spin_min_in_bin = spin_min + bin_index * spin_binwidth
        spin_max_in_bin = spin_min + (bin_index + 1) * spin_binwidth
        sample_id_in_prior = prior.loc[(prior["chif"] >= spin_min_in_bin) & (prior["chif"] <= spin_max_in_bin)].index
        if len(sample_id_in_prior) > 0:
            kick_posterior[sample_id_in_prior, "weights"] += 1 / len(sample_id_in_prior)
        else:
            LOGGER.warning("No samples in prior bin, not enough of samples.")

    LOGGER.debug(f"Computational time for kick estimation: {(time() - estimation_start_time):.1f} seconds.")
    if savecsv:
        if posterior_label is None:
            raise ValueError("posterior_label must not be empty if savecsv is True.")
        if output_dir is None:
            raise ValueError("output_dir must not be None if savecsv is True.")
        filepath = f"{output_dir}/{posterior_label}_posterior.csv"
        kick_posterior.to_csv(filepath, index=False)
        LOGGER.debug(f"Saved the estimated posterior to {filepath}.")
    kick_posterior = kick_posterior.set_index("weights")
    return kick_posterior
