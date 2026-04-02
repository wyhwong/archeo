import pandas as pd


def compute_bias_for_remnant_spin(df_samples: pd.DataFrame) -> float:
    """Compute the bias in remnant spin for a given set of samples."""

    return (df_samples["a_f"] - df_samples["spin_measure"]).sum() / df_samples.shape[0]


def compute_bias_for_remnant_mass(df_samples: pd.DataFrame) -> float:
    """Compute the bias in remnant mass for a given set of samples."""

    return (df_samples["m_f"] - df_samples["mass_measure"]).sum() / df_samples.shape[0]
