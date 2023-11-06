import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait

import env
import utils.posterior


def infer_parental_posterior(
    df: pd.DataFrame,
    label: str,
    spin_posterior: list[float],
    mass_posterior: list[float],
    mass_injection: bool,
    num_samples: int = 10,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """
    Infer the parental posterior from the posterior of the child parameters.

    Parameters
    ----------
    df : pd.DataFrame
        The prior of the child parameters.
    label : str
        The label of the child parameters.
    spin_posterior : list[float]
        The posterior of the spin parameter.
    mass_posterior : list[float]
        The posterior of the mass parameter.
    mass_injection : bool
        Whether the mass is injected.
    num_samples : int, optional
        The number of samples to draw from the prior per posterior sample, by default 10
    output_dir : str, optional
        The directory to save the posterior to, by default None

    Returns
    -------
    posterior : pd.DataFrame
        The posterior of the parental parameters.
    """
    sampler = utils.posterior.PosteriorSampler(df, mass_injection, num_samples)

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
