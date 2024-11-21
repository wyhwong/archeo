from archeo.constants import Fits
from archeo.schema import Domain, PriorConfig


PRECESSING_PRIOR = PriorConfig(
    n_samples=2000000,
    fits=Fits.NRSUR7DQ4REMNANT,
    is_spin_aligned=False,
    is_only_up_aligned_spin=False,
    spin=Domain(low=0.0, high=1.0),
    phi=Domain(low=0.0, high=2.0),
    theta=Domain(low=0.0, high=1.0),
    mass=Domain(low=5.0, high=65.0),
    mass_ratio=Domain(low=1.0, high=6.0),
    is_mahapatra=False,
)
