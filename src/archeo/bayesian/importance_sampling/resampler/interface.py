import numpy as np
import pandas as pd
from tqdm import tqdm

from archeo.bayesian.importance_sampling.resampler.assume_independence import ISDataAssumeIndependence
from archeo.bayesian.importance_sampling.resampler.generic import ISDataGeneric
from archeo.data_structures.annotation import Interface
from archeo.data_structures.bayesian.bayes_factor import BayesFactor
from archeo.utils.decorator import pre_release
from archeo.utils.logger import get_logger
from archeo.utils.parallel import multithread_run


LOGGER = get_logger(__name__)


class ImportanceSamplingData(ISDataGeneric, ISDataAssumeIndependence, Interface):
    """Importance sampling data for all resamplers"""

    assume_parameter_independence: bool = False

    @pre_release
    def get_likelihood_samples(self, random_state=42) -> np.ndarray:
        """Get samples for likelihood function"""

        if self.assume_parameter_independence:
            return self.get_likelihood_samples_1d(random_state=random_state)

        return self.get_likelihood_samples_dd(random_state=random_state)

    @pre_release
    def get_bayes_factor(self, bootstrapping: bool = False) -> float:
        """Compute the Bayes factor between two models

        NOTE: In this implementation, the likelihood function remains untouched.
        So that the Bayes factor is computed as the ratio of the new prior to the old prior.
        Details please check importance sampling.
        """

        if self.new_prior_samples.empty:
            return 0.0

        if self.assume_parameter_independence:
            return self.get_bayes_factor_1d(bootstrapping=bootstrapping)

        return self.get_bayes_factor_dd(bootstrapping=bootstrapping)

    @pre_release
    def sample_bayes_factor(self, n: int, is_parallel: bool = False, n_threads: int | None = None) -> BayesFactor:
        """Sample the Bayes factor for the importance sampling"""

        if self.new_prior_samples.empty:
            return BayesFactor(samples=[0.0] * n)

        if self.assume_parameter_independence:
            if is_parallel:
                return BayesFactor(
                    samples=multithread_run(
                        func=self.get_bayes_factor_1d,
                        input_kwargs=[{"bootstrapping": True} for _ in range(n)],
                        n_threads=n_threads,
                    )
                )
            return BayesFactor(samples=[self.get_bayes_factor_1d(bootstrapping=True) for _ in tqdm(range(n))])

        if is_parallel:
            return BayesFactor(
                samples=multithread_run(
                    func=self.get_bayes_factor_dd,
                    input_kwargs=[{"bootstrapping": True} for _ in range(n)],
                    n_threads=n_threads,
                )
            )
        return BayesFactor(samples=[self.get_bayes_factor_dd(bootstrapping=True) for _ in tqdm(range(n))])

    @pre_release
    def get_reweighted_samples(self, random_state=42) -> pd.DataFrame:
        """Get the reweighted samples for the importance sampling"""

        if self.assume_parameter_independence:
            return self.get_reweighted_samples_1d(random_state=random_state)

        return self.get_reweighted_samples_dd(random_state=random_state)
