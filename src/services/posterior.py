import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Optional

import env
import utils

logger = utils.logger.get_logger(logger_name="services|posterior")


def infer_parental_posterior(
    sampler: utils.posterior.PosteriorSampler,
    label: str,
    spin_posterior: list[float],
    mass_posterior: list[float],
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Infer the parental posterior from the posterior of the child parameters.

    Parameters
    ----------
    sampler : utils.posterior.PosteriorSampler
        The posterior sampler.
    label : str
        The label of the child parameters.
    spin_posterior : list[float]
        The posterior of the spin parameter.
    mass_posterior : list[float]
        The posterior of the mass parameter.
    output_dir : str, optional
        The directory to save the posterior to, by default None

    Returns
    -------
    posterior : pd.DataFrame
        The posterior of the parental parameters.
    """
    logger.info("Running the parental posterior inference... (%s)", label)

    with ProcessPoolExecutor(max_workers=env.MAX_WORKER) as Executor:
        futures = [
            Executor.submit(sampler.infer_parental_params, spin_measure, mass_measure)
            for spin_measure, mass_measure in zip(tqdm(spin_posterior), mass_posterior)
        ]
    wait(futures)

    posterior = pd.concat([future.result() for future in futures])
    if output_dir:
        filepath = f"{output_dir}/{label}_parental_params.h5"
        posterior.to_hdf(filepath, key="estimates", index=False)
    return posterior
