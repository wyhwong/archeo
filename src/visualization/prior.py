import pandas as pd
import seaborn as sns

import schemas.visualization
import visualization.base as base


def plot_prior_dist(df_prior: pd.DataFrame, output_dir=None, close=True) -> None:
    """
    Plot the distribution of the prior parameters.

    Parameters
    ----------

    df_prior : pd.DataFrame
        Prior.
    output_dir : str, optional
        Output directory.
    close : bool, optional
        Whether to close the figure.

    Returns
    -------
    None
    """
    labels = schemas.visualization.Labels("Distribution of remnant black-hole parameters")
    _, axes = base.initialize_plot(nrows=len(df_prior.columns), figsize=(16, 16), labels=labels)
    for index, column in enumerate(df_prior.columns):
        sns.histplot(df_prior[column], ax=axes[index], element="step", fill=False, stat="density")
    base.savefig_and_close("prior.png", output_dir, close)
