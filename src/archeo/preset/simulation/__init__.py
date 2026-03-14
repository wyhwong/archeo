from typing import Callable

from archeo.data_structures.physics.simulation import PipelineOutput
from archeo.preset.simulation.agnostic import (
    simulate_agnostic_aligned_spin_binaries,
    simulate_agnostic_precession_spin_binaries,
)
from archeo.preset.simulation.n_generation import (
    simulate_multi_generation_aligned_spin_binaries,
    simulate_multi_generation_precession_spin_binaries,
)
from archeo.preset.simulation.second_generation import (
    simulate_second_generation_aligned_spin_binaries,
    simulate_second_generation_precession_spin_binaries,
)
from archeo.utils.logger import get_logger


LOGGER = get_logger(__name__)

BINARY_GENERATION_PIPELINE_STORE = {
    "agnostic_precession_spin": simulate_agnostic_precession_spin_binaries,
    "agnostic_aligned_spin": simulate_agnostic_aligned_spin_binaries,
    "2g_precession_spin": simulate_second_generation_precession_spin_binaries,
    "2g_aligned_spin": simulate_second_generation_aligned_spin_binaries,
    "ng_precession_spin": simulate_multi_generation_precession_spin_binaries,
    "ng_aligned_spin": simulate_multi_generation_aligned_spin_binaries,
}

AVAILABLE_PIPELINES = list(BINARY_GENERATION_PIPELINE_STORE.keys())
CLI_USEABLE_PIPELINES = [
    "agnostic_precessing_spin",
    "agnostic_aligned_spin",
    "2g_precession_spin",
    "2g_aligned_spin",
]


def get_binary_generation_pipeline(name: str) -> Callable[..., PipelineOutput]:
    """Get the binary generation pipeline by name.

    Args:
        name (str): The name of the binary generation pipeline. Must be one of:
            - "agnostic_precessing_spin"
            - "agnostic_aligned_spin"
            - "2g_precessing_spin"
            - "2g_aligned_spin"
            - "ng_precessing_spin"
            - "ng_aligned_spin"

    Returns:
        Callable: The binary generation function corresponding to the given name.

    Raises:
        ValueError: If the provided name is not in the BINARY_STORE.
    """

    if name not in BINARY_GENERATION_PIPELINE_STORE:
        raise ValueError(f"Invalid binary generation pipeline name: {name}. Must be one of: {AVAILABLE_PIPELINES}")

    pipeline = BINARY_GENERATION_PIPELINE_STORE[name]
    LOGGER.info("Selected binary generation pipeline: %s", name)
    LOGGER.info("Pipeline function introduction: \n %s", pipeline.__doc__)

    return pipeline
