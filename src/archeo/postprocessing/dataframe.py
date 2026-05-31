import pandas as pd

from archeo.data_structures.bayesian.bayes_factor import BayesFactorCurveData, BayesFactorCurveMetadata
from archeo.data_structures.physics.simulation import BlackHoleMergers


def convert_simulated_binaries_to_dataframe(black_hole_mergers: BlackHoleMergers) -> pd.DataFrame:
    """Convert merger outputs into a flat dataframe schema.

    Args:
        black_hole_mergers (BlackHoleMergers): Sequence of
            ``(Binary, remnant_black_hole)`` tuples.

    Returns:
        pd.DataFrame: Dataframe containing parent properties, remnant properties,
        and derived binary quantities.
    """

    records = [
        {
            "m_1": binary.primary_black_hole.mass,
            "a_1": binary.primary_black_hole.spin_magnitude,
            "a_1x": binary.primary_black_hole.spin_vector[0],
            "a_1y": binary.primary_black_hole.spin_vector[1],
            "a_1z": binary.primary_black_hole.spin_vector[2],
            "v_1": binary.primary_black_hole.speed,
            "m_2": binary.secondary_black_hole.mass,
            "a_2": binary.secondary_black_hole.spin_magnitude,
            "a_2x": binary.secondary_black_hole.spin_vector[0],
            "a_2y": binary.secondary_black_hole.spin_vector[1],
            "a_2z": binary.secondary_black_hole.spin_vector[2],
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
    """Convert Bayes-factor curve objects to a tabular dataframe.

    Args:
        bayes_factor_curve_data (BayesFactorCurveData): Mapping from escape velocity
            to sampled Bayes-factor object.
        metadata (BayesFactorCurveMetadata): Curve-level metadata used for
            normalization and bookkeeping.

    Returns:
        pd.DataFrame: Dataframe containing per-velocity Bayes-factor summaries
        and metadata fields.
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
