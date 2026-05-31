import pandas as pd


def compute_bias_for_remnant_spin(df_samples: pd.DataFrame) -> float:
    """Compute mean estimation bias for remnant spin.

    Args:
        df_samples (pd.DataFrame): Samples containing ``a_f`` and ``spin_measure``.

    Returns:
        float: Mean value of ``a_f - spin_measure``.
    """

    return (df_samples["a_f"] - df_samples["spin_measure"]).sum() / df_samples.shape[0]


def compute_bias_for_remnant_mass(df_samples: pd.DataFrame) -> float:
    """Compute mean estimation bias for remnant mass.

    Args:
        df_samples (pd.DataFrame): Samples containing ``m_f`` and ``mass_measure``.

    Returns:
        float: Mean value of ``m_f - mass_measure``.
    """

    return (df_samples["m_f"] - df_samples["mass_measure"]).sum() / df_samples.shape[0]
