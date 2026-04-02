import pandas as pd

from archeo.postprocessing.eval_utils.bias import compute_bias_for_remnant_mass, compute_bias_for_remnant_spin
from archeo.postprocessing.eval_utils.kl import compute_kl_divergence_from_samples


def evaluate_ancestral_inference(df_samples: pd.DataFrame) -> dict[str, float]:
    """Evaluate the validity of the estimated posterior samples by computing:
    1. Estimation bias: the average difference between the estimated parameters and the true parameters
    2. KL divergence between the Gaussian fitted to the estimated samples and the Gaussian fitted to
       the true parameters.
    """

    bias_spin = compute_bias_for_remnant_spin(df_samples)
    bias_mass = compute_bias_for_remnant_mass(df_samples)

    mask = df_samples[["a_f", "m_f"]].notna().all(axis=1)
    # Here we need to apply the mask to the inferred samples.
    # The reason is that there could be no similar samples
    # in the ancestral prior, resulting in NaN values.
    # Those NaN values would cause the KL divergence computation to fail.
    kl_div = compute_kl_divergence_from_samples(
        df_samples.loc[mask, ["a_f", "m_f"]].values, df_samples[["spin_measure", "mass_measure"]].values
    )
    return {
        "bias_spin": bias_spin,
        "bias_mass": bias_mass,
        "kl_divergence": kl_div,
    }
