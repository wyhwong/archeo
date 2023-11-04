import p_tqdm
import pandas as pd

import utils.posterior


def infer_parental_posterior(
    df: pd.DataFrame,
    label: str,
    spin_posterior: list[float],
    mass_posterior: list[float],
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
    num_samples : int, optional
        The number of samples to draw from the prior per posterior sample, by default 10
    output_dir : str, optional
        The directory to save the posterior to, by default None

    Returns
    -------
    posterior : pd.DataFrame
        The posterior of the parental parameters.
    """
    sampler = utils.posterior.PosteriorSampler(df, num_samples)
    posterior = pd.concat(p_tqdm.p_map(sampler.infer_parental_params, spin_posterior, mass_posterior))
    if output_dir:
        filepath = f"{output_dir}/{label}_parental_params.h5"
        posterior.to_hdf(filepath, key="estimates", index=False)
    return posterior
