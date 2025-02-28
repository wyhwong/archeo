import archeo.logger
from archeo.preset.agnostic_aligned import AGNOSTIC_ALIGNED_SPIN_PRIOR
from archeo.preset.agnostic_precessing import AGNOSTIC_PRECESSING_SPIN_PRIOR
from archeo.preset.aligned import ALIGNED_SPIN_PRIOR
from archeo.preset.positively_aligned import POSITIVELY_ALIGNED_SPIN_PRIOR
from archeo.preset.precessing import PRECESSING_SPIN_PRIOR
from archeo.schema import PriorConfig


local_logger = archeo.logger.get_logger(__name__)


PRIOR_STORE = {
    "default": AGNOSTIC_PRECESSING_SPIN_PRIOR,
    "agnostic_precessing_spin": AGNOSTIC_PRECESSING_SPIN_PRIOR,
    "agnostic_aligned_spin": AGNOSTIC_ALIGNED_SPIN_PRIOR,
    "precessing_spin": PRECESSING_SPIN_PRIOR,
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

    if name not in PRIOR_STORE:
        msg = f"Prior {name} not found. "
        local_logger.error(msg)
        extra_msg = f"Available priors are: {list(PRIOR_STORE.keys())}"
        raise ValueError(msg + extra_msg)

    return PRIOR_STORE[name]
