import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Callable

import env
import utils
import schemas


logger = utils.get_logger(logger_name="services|prior")


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


def run_simulation(
    config: schemas.binary.BinaryConfig,
    is_mass_injected: bool,
    num_binaries: int,
    mass_ratio_from_pdf: Callable | None = None,
    mass_from_pdf: Callable | None = None,
    output_dir: str = "",
) -> pd.DataFrame:
    """
    Run a prior simulation.

    Parameters
    ----------
    config : schemas.binary.BinaryConfig
        Configuration of the binary generator.
    is_mass_injected : bool
        Whether to inject the masses
    num_binaries : int
        Number of binaries to simulate.
    mass_ratio_pdf : Callable, optional
        Mass ratio pdf, by default None
    mass_pdf : Callable, optional
        Mass pdf, by default None
    output_dir : str
        Output directory.

    Returns
    -------
    prior : pd.DataFrame
        Prior.
    """
    logger.info("Running the prior simulation...")
    logger.info("Number of binaries: %d", num_binaries)
    logger.info("Is mass injected: %s", is_mass_injected)

    global generator
    generator = utils.binary.BinaryGenerator(
        config, is_mass_injected, mass_ratio_from_pdf, mass_from_pdf
    )

    with ProcessPoolExecutor(max_workers=env.MAX_WORKER) as Executor:
        futures = [
            Executor.submit(_get_remnant_params, generator())
            for _ in tqdm(range(num_binaries))
        ]
    wait(futures)

    samples = [future.result() for future in futures]
    df = pd.DataFrame(samples, columns=["m1", "m2", "q", "mf", "vf", "chif"])
    df.to_csv(f"{output_dir}/prior.csv", index=False)
    return df
