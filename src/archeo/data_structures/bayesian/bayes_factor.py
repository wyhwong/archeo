from typing import TypeAlias

import numpy as np
from pydantic import BaseModel, NonNegativeFloat

from archeo.constants.bayesian import ASSUME_PARAMETER_INDEPENDENCE, DEFAULT_BINSIZE_MASS, DEFAULT_BINSIZE_SPIN


class BayesFactor(BaseModel, frozen=True):
    """Data class for Bayes factor with bootstrapping samples."""

    samples: list[float]

    def median(self) -> float:
        """Get the median of the samples."""

        return float(np.median(self.samples))

    def confidence_interval(self, percent: float = 90.0) -> tuple[float, float]:
        """Get the confidence interval of the samples."""

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
