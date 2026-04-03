import numpy as np
import pandas as pd
import pytest

from archeo.bayesian.ancestral_posterior import infer_ancestral_posterior_distribution


def _tiny_prior():
    return pd.DataFrame(
        {
            "m_f": [30.0, 40.0],
            "a_f": [0.2, 0.8],
            "m_1": [35.0, 45.0],
            "m_2": [20.0, 25.0],
            "k_f": [100.0, 200.0],
        }
    )


def test_infer_ancestral_posterior_length_mismatch():
    with pytest.raises(ValueError, match="must be the same"):
        infer_ancestral_posterior_distribution(
            df_binaries=_tiny_prior(),
            mass_posterior_samples=[30.0, 31.0],
            spin_posterior_samples=[0.2],
            n_workers=1,
        )


def test_infer_ancestral_posterior_no_matches_returns_rows():
    out = infer_ancestral_posterior_distribution(
        df_binaries=_tiny_prior(),
        mass_posterior_samples=[999.0, 888.0],
        spin_posterior_samples=[0.99, 0.98],
        binsize_mass=0.01,
        binsize_spin=0.001,
        n_workers=1,
    )
    assert len(out) == 2
    # logL should be -inf for no-likelihood matches
    assert np.isneginf(out["logL"]).all()
