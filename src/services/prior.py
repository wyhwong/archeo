import pandas as pd
import p_tqdm

import utils

logger = utils.logger.get_logger(logger_name="utils|prior")


def _get_binary(generator: utils.binary.BinaryGenerator) -> list[float]:
    """
    Get a binary.

    Parameters
    ----------
    generator : utils.binary.BinaryGenerator
        Binary generator.

    Returns
    -------
    binary : list[float]
        Parameters of the binary remnant.
    """
    return generator().get_remnant_params()


def run_simulation(generator: utils.binary.BinaryGenerator, num_binaries: int, output_dir: str) -> pd.DataFrame:
    """
    Run a prior simulation.

    Parameters
    ----------
    generator : utils.binary.BinaryGenerator
        Binary generator.
    num_binaries : int
        Number of binaries to simulate.
    output_dir : str
        Output directory.

    Returns
    -------
    prior : pd.DataFrame
        Prior.
    """
    prior = p_tqdm.p_map(_get_binary, p_tqdm.t_imap(utils.common.return_input, [generator] * num_binaries))
    df_prior = pd.DataFrame(prior, columns=["q", "mf", "vf", "chif"])
    df_prior.to_csv(f"{output_dir}/prior.csv", index=False)
    return df_prior
