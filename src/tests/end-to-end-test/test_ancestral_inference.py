import os

import numpy as np
import pandas as pd
import pytest

from archeo.bayesian.ancestral_posterior import infer_ancestral_posterior_distribution
from archeo.constants.physics import EscapeVelocity
from archeo.preset.simulation.agnostic import simulate_agnostic_aligned_spin_binaries


SAMPLE_SIZE = 50000


# Make a fixture, it is a dataframe, read the json from test_data/gw190521_lvk_subsampled.json
@pytest.fixture(name="gw190521_pe_samples")
def get_gw190521_pe_samples() -> pd.DataFrame:
    """Fixture that provides the posterior samples for GW190521 from the LVK subsampled dataset."""

    filepath = f"{os.path.dirname(os.path.dirname(__file__))}/test_data/gw190521_lvk_subsampled.json"
    return pd.read_json(filepath)


def test_gw190521_ancestral_inference(gw190521_pe_samples: pd.DataFrame):

    df_binaries, _ = simulate_agnostic_aligned_spin_binaries(size=SAMPLE_SIZE, n_workers=-1)
    df_bh1_ancestors = infer_ancestral_posterior_distribution(
        df_binaries=df_binaries,
        mass_posterior_samples=gw190521_pe_samples["mass_1_source"].values.tolist(),
        spin_posterior_samples=gw190521_pe_samples["a_1"].values.tolist(),
        n_workers=-1,
    )
    df_bh2_ancestors = infer_ancestral_posterior_distribution(
        df_binaries=df_binaries,
        mass_posterior_samples=gw190521_pe_samples["mass_2_source"].values.tolist(),
        spin_posterior_samples=gw190521_pe_samples["a_2"].values.tolist(),
        n_workers=-1,
    )

    # Minimum requirement: retieved samples has same number as the posterior samples
    assert len(df_bh1_ancestors) == len(gw190521_pe_samples)
    assert len(df_bh2_ancestors) == len(gw190521_pe_samples)

    # The following checks are based on https://arxiv.org/abs/2404.00720
    # Our paper on GW190521, note that we won't be able to 100% reproduce the results.
    # But the results should be roughly consistent with the paper.

    # Primary Black Hole results
    assert np.isclose(EscapeVelocity.GLOBULAR_CLUSTER.compute_p2g(df_bh1_ancestors), 13.5, atol=10.0)
    assert np.isclose(EscapeVelocity.MILKY_WAY.compute_p2g(df_bh1_ancestors), 60.5, atol=10.0)
    assert np.isclose(EscapeVelocity.NUCLEAR_STAR_CLUSTER.compute_p2g(df_bh1_ancestors), 60.5, atol=10.0)
    assert np.isclose(EscapeVelocity.ELLIPTICAL_GALAXY.compute_p2g(df_bh1_ancestors), 60.5, atol=10.0)

    assert np.isclose(df_bh1_ancestors["m_1"].median(), 64, atol=8.5)
    assert np.isclose(df_bh1_ancestors["m_2"].median(), 26, atol=6.5)
    assert np.isclose(df_bh1_ancestors["a_1"].median(), 0.66, atol=0.16)
    assert np.isclose(df_bh1_ancestors["a_2"].median(), 0.52, atol=0.22)
    assert np.isclose(df_bh1_ancestors["chi_eff"].median(), 0.15, atol=0.355)
    assert np.isclose(df_bh1_ancestors["k_f"].median(), 117, atol=50)

    # Secondary Black Hole results
    assert np.isclose(EscapeVelocity.GLOBULAR_CLUSTER.compute_p2g(df_bh2_ancestors), 25.5, atol=10.0)
    assert np.isclose(EscapeVelocity.MILKY_WAY.compute_p2g(df_bh2_ancestors), 95.1, atol=10.0)
    assert np.isclose(EscapeVelocity.NUCLEAR_STAR_CLUSTER.compute_p2g(df_bh2_ancestors), 95.1, atol=10.0)
    assert np.isclose(EscapeVelocity.ELLIPTICAL_GALAXY.compute_p2g(df_bh2_ancestors), 95.1, atol=10.0)

    assert np.isclose(df_bh2_ancestors["m_1"].median(), 47, atol=7.5)
    assert np.isclose(df_bh2_ancestors["m_2"].median(), 20, atol=5.0)
    assert np.isclose(df_bh2_ancestors["a_1"].median(), 0.67, atol=0.155)
    assert np.isclose(df_bh2_ancestors["a_2"].median(), 0.52, atol=0.22)
    assert np.isclose(df_bh2_ancestors["chi_eff"].median(), 0.23, atol=0.32)
    assert np.isclose(df_bh2_ancestors["k_f"].median(), 104, atol=44)
