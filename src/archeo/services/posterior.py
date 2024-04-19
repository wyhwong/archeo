from typing import Optional

import pandas as pd

import archeo.core.executor
import archeo.core.posterior
import archeo.logger


local_logger = archeo.logger.get_logger(__name__)


def infer_parental_posterior(
    sampler: archeo.core.posterior.sampler.PosteriorSampler,
    label: str,
    spin_posterior: list[float],
    mass_posterior: list[float],
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Infer the parental posterior.

    Args:
    -----
        sampler (archeo.core.posterior.sampler.PosteriorSampler):
            Posterior sampler.

        label (str):
            Label of the run.

        spin_posterior (list[float]):
            Spin posterior.

        mass_posterior (list[float]):
            Mass posterior.

        output_dir (Optional[str]):
            Output directory.

    Returns:
    -----
        posterior (pd.DataFrame):
            Parental posterior.
    """

    local_logger.info("Running the parental posterior inference... (%s)", label)

    executor = archeo.core.executor.MultiThreadExecutor()
    input_kwargs = [
        dict(spin_measure=spin_measure, mass_measure=mass_measure)
        for spin_measure, mass_measure in zip(spin_posterior, mass_posterior)
    ]
    results = executor.run(func=sampler.infer_parental_params, input_kwargs=input_kwargs)
    posterior = pd.concat(results)

    if output_dir:
        filepath = f"{output_dir}/{label}_parental_params.feather"
        posterior.to_feather(filepath)

    return posterior
