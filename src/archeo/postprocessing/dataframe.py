import pandas as pd

from archeo.data_structures.physics.binary import Binaries
from archeo.data_structures.physics.black_hole import BlackHoles


def convert_simulated_binaries_to_dataframe(binaries: Binaries, remnants: BlackHoles) -> pd.DataFrame:
    """Convert the simulated binaries and remnants to a pandas DataFrame.

    Args:
        binaries (Binaries): List of Binary objects representing the simulated binaries.
        remnants (BlackHoles): List of BlackHole objects representing the remnants of the mergers.

    Returns:
        pd.DataFrame: A DataFrame containing the properties of the binaries and their remnants.
    """

    records = []
    for binary, remnant in zip(binaries, remnants):
        records.append(
            {
                "m_1": binary.primary_black_hole.mass,
                "a_1": binary.primary_black_hole.spin_magnitude,
                "v_1": binary.primary_black_hole.speed,
                "m_2": binary.secondary_black_hole.mass,
                "a_2": binary.secondary_black_hole.spin_magnitude,
                "v_2": binary.secondary_black_hole.speed,
                "m_f": remnant.mass,
                "a_f": remnant.spin_magnitude,
                "v_f": remnant.speed,
                "chi_eff": binary.effective_spin,
                "chi_p": binary.precession_spin,
            }
        )
    return pd.DataFrame(records)
