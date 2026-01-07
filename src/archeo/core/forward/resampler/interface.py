import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from archeo.core.forward.resampler.assume_independence import ISDataAssumeIndependence
from archeo.core.forward.resampler.generic import ISDataGeneric
from archeo.schema import Interface
from archeo.utils.helper import pre_release


local_logger = logging.getLogger(__name__)


@dataclass
class ImportanceSamplingData(Interface, ISDataGeneric, ISDataAssumeIndependence):
    """Importance sampling data for all resamplers"""

    assume_parameter_independence: bool = False

    @pre_release
    def get_likelihood_samples(self, random_state=42, ztol=1e-8) -> np.ndarray:
        """Get samples for likelihood function"""

        if self.assume_parameter_independence:
            return self.get_likelihood_samples_1d(random_state=random_state, ztol=ztol)

        return self.get_likelihood_samples_dd(random_state=random_state, ztol=ztol)

    @pre_release
    def get_weights(self, col_name: str, ztol=1e-8) -> np.ndarray:
        """Get the weights for the importance sampling"""

        if self.assume_parameter_independence:
            return self.get_weights_1d(col_name=col_name, ztol=ztol)

        return self.get_weights_dd(col_name=col_name, ztol=ztol)

    @pre_release
    def get_bayes_factor(self, ztol=1e-8) -> float:
        """Compute the Bayes factor between two models

        NOTE: In this implementation, the likelihood function remains untouched.
        So that the Bayes factor is computed as the ratio of the new prior to the old prior.
        Details please check importance sampling.
        """

        if self.assume_parameter_independence:
            return self.get_bayes_factor_1d(ztol=ztol)

        return self.get_bayes_factor_dd(ztol=ztol)

    @pre_release
    def get_reweighted_samples(self, ztol=1e-8, random_state=42) -> pd.DataFrame:
        """Get the reweighted samples for the importance sampling"""

        if self.assume_parameter_independence:
            return self.get_reweighted_samples_1d(ztol=ztol, random_state=random_state)

        return self.get_reweighted_samples_dd(ztol=ztol, random_state=random_state)
