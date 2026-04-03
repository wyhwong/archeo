import numpy as np
import pandas as pd
import pytest

from archeo.data_structures.bayesian.bayes_factor import BayesFactor, BayesFactorCurveMetadata
from archeo.data_structures.physics.binary import Binary
from archeo.data_structures.physics.black_hole import BlackHole
from archeo.postprocessing.dataframe import (
    convert_bayes_factor_curve_to_dataframe,
    convert_simulated_binaries_to_dataframe,
)
from archeo.postprocessing.eval_utils.bias import (
    compute_bias_for_remnant_mass,
    compute_bias_for_remnant_spin,
)
from archeo.postprocessing.eval_utils.kl import compute_kl_divergence_from_samples
from archeo.postprocessing.evaluation import evaluate_ancestral_inference


def _make_merger():
    b1 = BlackHole(mass=30.0, spin_magnitude=0.4, spin_vector=(0.1, 0.2, 0.3), speed=0.0)
    b2 = BlackHole(mass=20.0, spin_magnitude=0.3, spin_vector=(0.0, 0.1, -0.2), speed=0.0)
    bin_ = Binary(primary_black_hole=b1, secondary_black_hole=b2)
    rem = BlackHole(mass=47.0, spin_magnitude=0.7, spin_vector=(0.0, 0.0, 0.7), speed=120.0)
    return (bin_, rem)


def test_convert_simulated_binaries_to_dataframe():
    df = convert_simulated_binaries_to_dataframe([_make_merger(), _make_merger()])
    expected_cols = {"m_1", "m_2", "m_f", "a_f", "k_f", "chi_eff", "chi_p", "q"}
    assert expected_cols.issubset(df.columns)
    assert len(df) == 2


def test_convert_bayes_factor_curve_to_dataframe():
    data = {
        50.0: BayesFactor(samples=[1.0, 2.0, 3.0]),
        100.0: BayesFactor(samples=[2.0, 3.0, 4.0]),
    }
    meta = BayesFactorCurveMetadata(reference_candidate_name="ref", reference_bayes_factor=2.0)
    df = convert_bayes_factor_curve_to_dataframe(data, meta)
    assert {"v_esc", "bayes_factor_low", "bayes_factor_high", "bayes_factor_median"}.issubset(df.columns)
    assert len(df) == 2


def test_bias_functions():
    df = pd.DataFrame(
        {"a_f": [0.6, 0.8], "spin_measure": [0.5, 0.7], "m_f": [60.0, 62.0], "mass_measure": [58.0, 60.0]}
    )
    assert np.isclose(compute_bias_for_remnant_spin(df), 0.1)
    assert np.isclose(compute_bias_for_remnant_mass(df), 2.0)


def test_kl_divergence_dimension_mismatch():
    X = np.random.normal(size=(100, 2))
    Y = np.random.normal(size=(100, 3))
    with pytest.raises(ValueError, match="same dimension"):
        compute_kl_divergence_from_samples(X, Y)


def test_kl_divergence_non_negative():
    X = np.random.normal(size=(200, 2))
    Y = np.random.normal(loc=0.5, size=(220, 2))
    kl = compute_kl_divergence_from_samples(X, Y)
    assert np.isfinite(kl)


def test_evaluate_ancestral_inference_masking():
    df = pd.DataFrame(
        {
            "a_f": [0.5, np.nan, 0.7],
            "m_f": [50.0, 60.0, np.nan],
            "spin_measure": [0.45, 0.55, 0.65],
            "mass_measure": [49.0, 59.0, 69.0],
        }
    )
    out = evaluate_ancestral_inference(df)
    assert {"bias_spin", "bias_mass", "kl_divergence"} == set(out.keys())
