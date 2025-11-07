import archeo.logger
from archeo.preset.agnostic import (
    AGNOSTIC_ALIGNED_SPIN_PRIOR,
    AGNOSTIC_PRECESSING_SPIN_PRIOR,
)
from archeo.preset.bh1g import (
    ALIGNED_SPIN_1G1G_PRIOR,
    POSITIVELY_ALIGNED_SPIN_1G1G_PRIOR,
    PRECESSING_SPIN_1G1G_PRIOR,
)
from archeo.preset.quick import TINY_ALIGNED_SPIN_PRIOR
from archeo.schema import PriorConfig


local_logger = archeo.logger.get_logger(__name__)


PRIOR_STORE = {
    "default": TINY_ALIGNED_SPIN_PRIOR,
    "tiny_aligned_spin": TINY_ALIGNED_SPIN_PRIOR,
    "agnostic_precessing_spin": AGNOSTIC_PRECESSING_SPIN_PRIOR,
    "agnostic_aligned_spin": AGNOSTIC_ALIGNED_SPIN_PRIOR,
    "precessing_spin": PRECESSING_SPIN_1G1G_PRIOR,
    "aligned_spin": ALIGNED_SPIN_1G1G_PRIOR,
    "positively_aligned_spin": POSITIVELY_ALIGNED_SPIN_1G1G_PRIOR,
}


def get_prior_config(name: str = "default") -> PriorConfig:
    """Get a prior configuration.
    NOTE: By default, it returns the precessing prior.

    Args:
        name (str): The name of the prior.

    Returns:
        prior_config (PriorConfig): The prior configuration.
    """

    if name not in PRIOR_STORE:
        msg = f"Prior {name} not found. "
        local_logger.error(msg)
        extra_msg = f"Available priors are: {list(PRIOR_STORE.keys())}"
        raise ValueError(msg + extra_msg)

    return PRIOR_STORE[name]
