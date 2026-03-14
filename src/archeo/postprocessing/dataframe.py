import pandas as pd

from archeo.data_structures.bayesian.bayes_factor import BayesFactorCurveData, BayesFactorCurveMetadata
from archeo.data_structures.physics.simulation import BlackHoleMergers


def convert_simulated_binaries_to_dataframe(black_hole_mergers: BlackHoleMergers) -> pd.DataFrame:
    """Convert the simulated binaries and remnants to a pandas DataFrame.

    Args:
        black_hole_mergers (BlackHoleMergers):
            List of tuples containing the binaries and their resulting black holes.

    Returns:
        pd.DataFrame: A DataFrame containing the properties of the binaries and their remnants.
    """

    records = [
        {
            "m_1": binary.primary_black_hole.mass,
            "a_1": binary.primary_black_hole.spin_magnitude,
            "v_1": binary.primary_black_hole.speed,
            "m_2": binary.secondary_black_hole.mass,
            "a_2": binary.secondary_black_hole.spin_magnitude,
            "v_2": binary.secondary_black_hole.speed,
            "m_f": remnant.mass,
            "a_f": remnant.spin_magnitude,
            "k_f": remnant.speed,
            "chi_eff": binary.effective_spin,
            "chi_p": binary.precession_spin,
            "q": binary.mass_ratio,
        }
        for binary, remnant in black_hole_mergers
    ]
    return pd.DataFrame(records)


def convert_bayes_factor_curve_to_dataframe(
    bayes_factor_curve_data: BayesFactorCurveData, metadata: BayesFactorCurveMetadata
) -> pd.DataFrame:
    """Convert the Bayes factor curve data to a pandas DataFrame.

    Args:
        bayes_factor_curve_data (BayesFactorCurveData):
            A dictionary mapping escape velocities to their corresponding Bayes factors.
        metadata (BayesFactorCurveMetadata):
            Metadata for the Bayes factor curve.

    Returns:
        pd.DataFrame: A DataFrame containing
            - the escape velocities,
            - Bayes factor confidence intervals,
            - Reference candidate name,
            - Reference Bayes factor,
            - Bin size (spin),
            - Bin size (mass).
    """

    records = [
        {
            "v_esc": v_esc,
            "bayes_factor_low": bayes_factor.confidence_interval()[0] / metadata.reference_bayes_factor,
            "bayes_factor_high": bayes_factor.confidence_interval()[1] / metadata.reference_bayes_factor,
            "bayes_factor_median": bayes_factor.median() / metadata.reference_bayes_factor,
            "bayes_factor_samples": bayes_factor.samples,
            "reference_candidate_name": metadata.reference_candidate_name,
            "reference_bayes_factor": metadata.reference_bayes_factor,
            "binsize_spin": metadata.binsize_spin,
            "binsize_mass": metadata.binsize_mass,
        }
        for v_esc, bayes_factor in bayes_factor_curve_data.items()
    ]
    return pd.DataFrame(records)
