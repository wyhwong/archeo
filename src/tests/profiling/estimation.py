import os

import pandas as pd
import pyinstrument

from archeo import infer_ancestral_posterior_distribution
from archeo.preset.simulation import simulate_second_generation_aligned_spin_binaries


@pyinstrument.profile()
def profile_infer_ancestral_posterior_distribution():
    """Profile the infer_ancestral_posterior_distribution function."""

    test_data_dir = f"{os.path.dirname(os.path.dirname(__file__))}/test_data"
    df_binaries, _ = simulate_second_generation_aligned_spin_binaries(size=10000, n_workers=-1)
    df_pe_samples = pd.read_json(f"{test_data_dir}/gw190521_lvk_subsampled.json")

    infer_ancestral_posterior_distribution(
        df_binaries=df_binaries,
        mass_posterior_samples=df_pe_samples["mass_1_source"].tolist(),
        spin_posterior_samples=df_pe_samples["a_1"].tolist(),
        n_workers=1,
    )


@pyinstrument.profile()
def profile_infer_ancestral_posterior_distribution_parallel():
    """Profile the infer_ancestral_posterior_distribution function."""

    test_data_dir = f"{os.path.dirname(os.path.dirname(__file__))}/test_data"
    df_binaries, _ = simulate_second_generation_aligned_spin_binaries(size=10000, n_workers=-1)
    df_pe_samples = pd.read_json(f"{test_data_dir}/gw190521_lvk_subsampled.json")

    infer_ancestral_posterior_distribution(
        df_binaries=df_binaries,
        mass_posterior_samples=df_pe_samples["mass_1_source"].tolist(),
        spin_posterior_samples=df_pe_samples["a_1"].tolist(),
        n_workers=-1,
    )


if __name__ == "__main__":
    profile_infer_ancestral_posterior_distribution()
    profile_infer_ancestral_posterior_distribution_parallel()
