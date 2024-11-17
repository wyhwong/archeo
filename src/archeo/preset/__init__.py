import archeo.logger
from archeo.schema import PriorConfig
from archeo.preset.aligned_spin import ALIGNED_SPIN_PRIOR
from archeo.preset.precessing import PRECESSING_PRIOR

local_logger = archeo.logger.get_logger(__name__)


PRIOR_STORE = {
    "aligned_spin": ALIGNED_SPIN_PRIOR,
    "precessing": PRECESSING_PRIOR,
}


def get_prior(name: str) -> PriorConfig:
    """
    Get a prior function.

    Args:
    -----
        name (str):
            The name of the prior.

    Returns:
    -----
        prior (Callable):
            The prior function.
    """

    if name not in PRIOR_STORE:
        msg = f"Prior {name} not found."
        local_logger.error(msg)
        raise ValueError(msg)

    return PRIOR_STORE[name]
