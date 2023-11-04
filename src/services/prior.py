import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait

import env
import utils
import schemas

logger = utils.logger.get_logger(logger_name="utils|prior")


def _get_remnant_params(binary: schemas.binary.Binary) -> list[float]:
    """
    Get a binary.

    Parameters
    ----------
    generator : utils.binary.BinaryGenerator
        Binary generator.

    Returns
    -------
    binary : list[float]
        Parameters of the binary remnant.
    """
    return binary.get_remnant_params(generator.config.fits)


def run_simulation(config: schemas.binary.BinaryConfig, num_binaries: int, output_dir: str) -> pd.DataFrame:
    """
    Run a prior simulation.

    Parameters
    ----------
    generator : utils.binary.BinaryGenerator
        Binary generator.
    num_binaries : int
        Number of binaries to simulate.
    output_dir : str
        Output directory.

    Returns
    -------
    prior : pd.DataFrame
        Prior.
    """
    global generator
    generator = utils.binary.BinaryGenerator(config)

    with ProcessPoolExecutor(max_workers=env.MAX_WORKER) as Executor:
        futures = [Executor.submit(_get_remnant_params, generator()) for _ in tqdm(range(num_binaries))]
    wait(futures)

    prior_samples = [future.result() for future in futures]
    df_prior = pd.DataFrame(prior_samples, columns=["q", "mf", "vf", "chif"])
    df_prior.to_csv(f"{output_dir}/prior.csv", index=False)
    return df_prior
