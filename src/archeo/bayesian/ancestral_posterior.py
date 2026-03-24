import numpy as np
import pandas as pd

from archeo.constants.bayesian import DEFAULT_BINSIZE_MASS, DEFAULT_BINSIZE_SPIN
from archeo.utils.parallel import get_n_workers, multiprocess_run, multithread_run


def _retrieve_sample(
    df_binaries: pd.DataFrame,
    mass_measure: float,
    spin_measure: float,
    binsize_mass: float = DEFAULT_BINSIZE_MASS,
    binsize_spin: float = DEFAULT_BINSIZE_SPIN,
) -> pd.DataFrame:
    """Retrieve the samples from data frame of binaries"""

    # Find the possible samples in the prior
    possible_samples = df_binaries.loc[
        ((df_binaries["m_f"] - mass_measure).abs() <= binsize_mass / 2)
        & ((df_binaries["a_f"] - spin_measure).abs() <= binsize_spin / 2)
    ]
    likelihood = len(possible_samples) / len(df_binaries)

    # Sample n_sample samples from the possible samples
    if possible_samples.empty:
        possible_samples = pd.DataFrame(index=[0], columns=possible_samples.columns)

    sample = possible_samples.sample(1)
    sample["logL"] = np.log(likelihood)
    sample["spin_measure"] = spin_measure
    sample["mass_measure"] = mass_measure

    return sample


def infer_ancestral_posterior_distribution(
    df_binaries: pd.DataFrame,
    mass_posterior_samples: list[float],
    spin_posterior_samples: list[float],
    binsize_mass: float = DEFAULT_BINSIZE_MASS,
    binsize_spin: float = DEFAULT_BINSIZE_SPIN,
    random_state: int = 42,
    n_workers: int = 1,
) -> pd.DataFrame:

    if len(mass_posterior_samples) != len(spin_posterior_samples):
        raise ValueError("The number of mass and spin posterior samples must be the same.")

    n_workers = get_n_workers(n_workers)

    if n_workers == 1:
        np.random.seed(random_state)

        return pd.concat(
            multithread_run(
                func=_retrieve_sample,
                input_kwargs=[
                    {
                        "df_binaries": df_binaries,
                        "mass_measure": mass_measure,
                        "spin_measure": spin_measure,
                        "binsize_mass": binsize_mass,
                        "binsize_spin": binsize_spin,
                    }
                    for mass_measure, spin_measure in zip(mass_posterior_samples, spin_posterior_samples)
                ],
            ),
            ignore_index=True,
        )

    # Separate the samples into chunks for parallel processing
    n_samples = len(mass_posterior_samples)
    chunk_size = (n_samples + n_workers - 1) // n_workers  # Calculate the chunk size
    spin_measure_chunks = [spin_posterior_samples[i : i + chunk_size] for i in range(0, n_samples, chunk_size)]
    mass_measure_chunks = [mass_posterior_samples[i : i + chunk_size] for i in range(0, n_samples, chunk_size)]

    # Process each chunk in parallel
    results = multiprocess_run(
        func=infer_ancestral_posterior_distribution,
        input_kwargs=[
            {
                "df_binaries": df_binaries,
                "mass_posterior_samples": mass_measure_chunk,
                "spin_posterior_samples": spin_measure_chunk,
                "binsize_mass": binsize_mass,
                "binsize_spin": binsize_spin,
                "random_state": random_state + i,
            }
            for i, (spin_measure_chunk, mass_measure_chunk) in enumerate(zip(spin_measure_chunks, mass_measure_chunks))
        ],
        n_processes=n_workers,
    )
    return pd.concat(results, ignore_index=True)
