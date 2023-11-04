import h5py
import pandas as pd

import utils.common


def read_posterior_from_json(filepath: str) -> dict:
    """
    Read the posterior of parameter estimation from a json file.

    Parameters
    ----------
    filepath : str
        Path to the json file.

    Returns
    -------
    posterior : dict
        Dictionary containing the posterior.
    """
    return utils.common.read_dict_from_json(filepath=filepath)["posterior"]["content"]


def read_posterior_from_h5(filepath: str, fits="NRSur7dq4") -> dict:
    """
    Read the posterior of parameter estimation from a h5 file.

    Parameters
    ----------
    filepath : str
        Path to the h5 file.

    fits : str
        Name of the waveform model.
    """
    return h5py.File(filepath, "r")[fits]["posterior_samples"]


class PosteriorSampler:
    """
    Posterior sampler for parameter estimation.
    """

    def __init__(self, df: pd.DataFrame, num_samples: int) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame
            The prior dataframe.
        num_samples : int
            The number of samples to be returned
        """
        self.num_samples = num_samples
        self.prior = df

    def infer_parental_params(self, spin_measure: float, mass_measure: float) -> pd.DataFrame:
        """
        Infer the parental parameters from the posterior of the child parameters.

        Parameters
        ----------
        spin_measure : float
            The spin measurement of the remnant
        mass_measure : float
            The mass measurement of the remnant

        Returns
        -------
        samples : pd.DataFrame
            The samples of the parental mass
        """
        samples = self.prior.iloc[(self.prior["chif"] - spin_measure).abs().argsort()[: self.num_samples]]
        samples["m_p1"] = mass_measure / samples["mf"] * samples["q"] / (1 + samples["q"])
        samples["m_p2"] = mass_measure / samples["mf"] / (1 + samples["q"])
        samples["mf"] = mass_measure
        return samples
