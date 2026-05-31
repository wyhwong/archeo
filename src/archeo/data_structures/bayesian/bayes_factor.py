from typing import TypeAlias

import numpy as np
from pydantic import BaseModel, NonNegativeFloat

from archeo.constants.bayesian import ASSUME_PARAMETER_INDEPENDENCE, DEFAULT_BINSIZE_MASS, DEFAULT_BINSIZE_SPIN


class BayesFactor(BaseModel, frozen=True):
    """Bootstrap-based Bayes-factor summary container.

    Stores Bayes-factor samples and provides robust summary statistics such as
    median and equal-tail confidence intervals.
    """

    samples: list[float]

    def median(self) -> float:
        """Return median Bayes factor across bootstrap samples.

        Returns:
            float: Median value.
        """

        return float(np.median(self.samples))

    def confidence_interval(self, percent: float = 90.0) -> tuple[float, float]:
        """Compute equal-tail confidence interval from bootstrap samples.

        Args:
            percent (float): Credible interval mass in percent.

        Returns:
            tuple[float, float]: Lower and upper interval bounds.
        """

        lower_percentile = (100 - percent) / 2
        upper_percentile = 100 - lower_percentile
        return (
            float(np.quantile(self.samples, lower_percentile / 100)),
            float(np.quantile(self.samples, upper_percentile / 100)),
        )


BayesFactorCurveData: TypeAlias = dict[float, BayesFactor]


class BayesFactorCurveMetadata(BaseModel, frozen=True):
    """Metadata for the Bayes factor curve."""

    reference_candidate_name: str = "original"
    reference_bayes_factor: NonNegativeFloat = 1.0
    binsize_spin: float = DEFAULT_BINSIZE_SPIN
    binsize_mass: float = DEFAULT_BINSIZE_MASS
    assume_parameter_independence: bool = ASSUME_PARAMETER_INDEPENDENCE
