import numpy as np

from archeo.data_structures.bayesian.bayes_factor import BayesFactor, BayesFactorCurveMetadata
from archeo.data_structures.physics.binary import Binary
from archeo.data_structures.physics.black_hole import BlackHole
from archeo.postprocessing.dataframe import (
    convert_bayes_factor_curve_to_dataframe,
    convert_simulated_binaries_to_dataframe,
)


def test_convert_simulated_binaries_to_dataframe_preserves_vector_components():
    p = BlackHole(mass=30.0, spin_magnitude=0.5, spin_vector=(0.1, 0.2, 0.3), speed=10.0)
    s = BlackHole(mass=20.0, spin_magnitude=0.4, spin_vector=(0.4, 0.5, 0.6), speed=20.0)
    b = Binary(primary_black_hole=p, secondary_black_hole=s)
    rem = BlackHole(mass=47.0, spin_magnitude=0.7, spin_vector=(0.0, 0.0, 0.7), speed=120.0)

    df = convert_simulated_binaries_to_dataframe([(b, rem)])
    row = df.iloc[0]

    assert np.isclose(row["a_1x"], 0.1)
    assert np.isclose(row["a_1y"], 0.2)
    assert np.isclose(row["a_1z"], 0.3)
    assert np.isclose(row["a_2x"], 0.4)
    assert np.isclose(row["a_2y"], 0.5)
    assert np.isclose(row["a_2z"], 0.6)
    assert np.isclose(row["m_f"], 47.0)
    assert np.isclose(row["k_f"], 120.0)


def test_convert_bayes_factor_curve_dataframe_scales_by_reference_bf():
    curve = {100.0: BayesFactor(samples=[2.0, 4.0, 6.0])}
    meta = BayesFactorCurveMetadata(reference_candidate_name="ref", reference_bayes_factor=2.0)

    df = convert_bayes_factor_curve_to_dataframe(curve, meta)
    row = df.iloc[0]

    assert row["reference_candidate_name"] == "ref"
    assert np.isclose(row["bayes_factor_median"], 2.0)  # median(2,4,6)=4 then /2
    assert isinstance(row["bayes_factor_samples"], list)
