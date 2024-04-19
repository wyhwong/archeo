from typing import Callable, Optional

import pandas as pd
import surfinBH
from tqdm import tqdm

import archeo.core.executor
import archeo.core.prior.binary
import archeo.core.prior.simulation
import archeo.logger
import archeo.schemas.binary


local_logger = archeo.logger.get_logger(__name__)


def _get_remnant_params(
    binary: archeo.schemas.binary.Binary,
    fits: surfinBH.surfinBH.SurFinBH,
) -> dict[str, float]:
    """
    Get remnant parameters.

    Args:
    -----
        binary (archeo.schemas.binary.Binary):
            The binary to simulate.

        fits (surfinBH.surfinBH.SurFinBH):
            Loaded fits.

    Returns:
    -----
        remnant_params (list[float]):
            The remnant parameters.
    """

    return archeo.core.prior.simulation.simulate_remnant(binary, fits)


def run_simulation(
    fits: archeo.schemas.binary.Fits,
    settings: archeo.schemas.binary.BinarySettings,
    is_mass_injected: bool,
    num_binaries: int,
    mass_ratio_from_pdf: Optional[Callable] = None,
    mass_from_pdf: Optional[Callable] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run prior simulation.

    Args:
    -----
        fits (archeo.schemas.binary.Fits):
            The fits to load. The available fits are:
            - NRSur3dq8Remnant: non precessing BHs with mass ratio<=8, anti-/aligned spin <= 0.8
            - NRSur7dq4Remnant: precessing BHs with mass ratio<=4, generic spin <= 0.8
            - surfinBH7dq2: precessing BHs with mass ratio <= 2, generic spin <= 0.8

        settings (archeo.schemas.binary.BinarySettings):
            The binary settings.

        is_mass_injected (bool):
            Whether to inject mass.

        num_binaries (int):
            The number of binaries to simulate.

        mass_ratio_from_pdf (Optional[Callable]):
            The function to generate mass ratio from pdf.

        mass_from_pdf (Optional[Callable]):
            The function to generate mass from pdf.

        output_dir (Optional[str]):
            The output directory to save the results.

    Returns:
    -----
        df (pd.DataFrame):
            The simulated binaries.
    """

    local_logger.info(
        "Run %d merger simulation... settings: %s",
        num_binaries,
        settings,
    )

    fits = archeo.core.prior.simulation.load_fits(fits)

    local_logger.info("Generating binaries...")
    generator = archeo.core.prior.binary.BinaryGenerator(
        settings=settings,
        is_mass_injected=is_mass_injected,
        mass_from_pdf=mass_from_pdf,
        mass_ratio_from_pdf=mass_ratio_from_pdf,
    )
    input_kwargs = [{"binary": generator(), "fits": fits} for _ in tqdm(range(num_binaries))]

    local_logger.info("Running simulation...")
    executor = archeo.core.executor.MultiThreadExecutor()
    samples = executor.run(_get_remnant_params, input_kwargs)
    df = pd.DataFrame(samples)

    if output_dir:
        local_logger.info("Saving results to feather file...")
        df.to_feather(f"{output_dir}/prior.feather")

    return df
