from archeo.schema import Domain, PriorConfig


TINY_ALIGNED_SPIN_PRIOR = PriorConfig(
    n_samples=5000,
    is_spin_aligned=True,
    is_only_up_aligned_spin=False,
    a_1=Domain(low=0.0, high=1.0),  # unit: dimensionless
    a_2=Domain(low=0.0, high=1.0),  # unit: dimensionless
    phi_1=Domain(low=0.0, high=2.0),  # unit: pi
    phi_2=Domain(low=0.0, high=2.0),  # unit: pi
    theta_1=Domain(low=0.0, high=1.0),  # unit: pi
    theta_2=Domain(low=0.0, high=1.0),  # unit: pi
    m_1=Domain(low=5.0, high=200.0),  # unit: solar mass
    m_2=Domain(low=5.0, high=200.0),  # unit: solar mass
    mass_ratio=Domain(low=1.0, high=6.0),  # unit: dimensionless
    is_mahapatra_mass_func=False,
    is_uniform_in_mass_ratio=False,
    is_masses_swappable=True,
)
