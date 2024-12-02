import archeo.logger
from archeo.preset.aligned import ALIGNED_SPIN_PRIOR
from archeo.preset.positively_aligned import POSITIVELY_ALIGNED_SPIN_PRIOR
from archeo.preset.precessing import PRECESSING_PRIOR
from archeo.schema import PriorConfig


local_logger = archeo.logger.get_logger(__name__)


PRIOR_STORE = {
    "precessing": PRECESSING_PRIOR,
    "aligned_spin": ALIGNED_SPIN_PRIOR,
    "positively_aligned_spin": POSITIVELY_ALIGNED_SPIN_PRIOR,
}


def get_prior_config(name: str = "default") -> PriorConfig:
    """Get a prior configuration.
    NOTE: By default, it returns the precessing prior.

    Args:
        name (str): The name of the prior.

    Returns:
        prior_config (PriorConfig): The prior configuration.
    """

    if name == "default":
        return PRIOR_STORE["precessing"]

    if name not in PRIOR_STORE:
        msg = f"Prior {name} not found."
        local_logger.error(msg)
        raise ValueError(msg)

    return PRIOR_STORE[name]
