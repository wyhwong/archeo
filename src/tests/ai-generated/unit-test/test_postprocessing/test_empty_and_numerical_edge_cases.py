import numpy as np
import pandas as pd
import pytest

from archeo.data_structures.bayesian.bayes_factor import BayesFactorCurveMetadata
from archeo.postprocessing.dataframe import (
    convert_bayes_factor_curve_to_dataframe,
    convert_simulated_binaries_to_dataframe,
)
from archeo.postprocessing.eval_utils.kl import compute_kl_divergence_from_samples
from archeo.postprocessing.evaluation import evaluate_ancestral_inference


def test_convert_simulated_binaries_to_dataframe_empty_input():
    df = convert_simulated_binaries_to_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_convert_bayes_factor_curve_to_dataframe_empty_input():
    df = convert_bayes_factor_curve_to_dataframe({}, BayesFactorCurveMetadata())
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_compute_kl_divergence_raises_for_singular_covariance():
    X = np.ones((10, 2))
    Y = np.ones((12, 2))
    with pytest.raises(np.linalg.LinAlgError):
        compute_kl_divergence_from_samples(X, Y)


def test_evaluate_ancestral_inference_returns_nan_kl_when_all_inferred_are_unmapped():
    df = pd.DataFrame(
        {
            "a_f": [np.nan, np.nan],
            "m_f": [np.nan, np.nan],
            "spin_measure": [0.4, 0.5],
            "mass_measure": [50.0, 60.0],
        }
    )
    out = evaluate_ancestral_inference(df)
    assert "bias_spin" in out and "bias_mass" in out and "kl_divergence" in out
    assert np.isnan(out["kl_divergence"])
