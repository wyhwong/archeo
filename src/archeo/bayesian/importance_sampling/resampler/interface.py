import numpy as np
import pandas as pd
from tqdm import tqdm

from archeo.bayesian.importance_sampling.resampler.assume_independence import ISDataAssumeIndependence
from archeo.bayesian.importance_sampling.resampler.generic import ISDataGeneric
from archeo.data_structures.bayesian.bayes_factor import BayesFactor
from archeo.data_structures.type_alias import Interface
from archeo.utils.decorator import pre_release
from archeo.utils.logger import get_logger
from archeo.utils.parallel import multithread_run


LOGGER = get_logger(__name__)


class ImportanceSamplingData(ISDataGeneric, ISDataAssumeIndependence, Interface):
    """Importance sampling data for all resamplers"""

    assume_parameter_independence: bool = False

    @pre_release
    def get_likelihood_samples(self, random_state=42) -> np.ndarray:
        """Dispatch likelihood-resampling method by model assumption.

        Args:
            random_state (int): Random seed for weighted sampling.

        Returns:
            np.ndarray: Resampled likelihood-like samples.
        """

        if self.assume_parameter_independence:
            return self.get_likelihood_samples_1d(random_state=random_state)

        return self.get_likelihood_samples_dd(random_state=random_state)

    @pre_release
    def get_bayes_factor(self, bootstrapping: bool = False) -> float:
        """Dispatch Bayes-factor computation by model assumption.

        Args:
            bootstrapping (bool): If ``True``, bootstrap sample sets before estimation.

        Returns:
            float: Bayes-factor estimate. Returns 0 if candidate prior is empty.
        """

        # NOTE: In this implementation, the likelihood function remains untouched.
        # So that the Bayes factor is computed as the ratio of the new prior to the old prior.
        # Details please check importance sampling.

        if self.new_prior_samples.empty:
            return 0.0

        if self.assume_parameter_independence:
            return self.get_bayes_factor_1d(bootstrapping=bootstrapping)

        return self.get_bayes_factor_dd(bootstrapping=bootstrapping)

    @pre_release
    def sample_bayes_factor(self, n: int, is_parallel: bool = False, n_threads: int | None = None) -> BayesFactor:
        """Draw bootstrap samples of the Bayes factor.

        Args:
            n (int): Number of bootstrap draws.
            is_parallel (bool): If ``True``, compute draws in parallel threads.
            n_threads (int | None): Optional thread count for parallel mode.

        Returns:
            BayesFactor: Container of sampled Bayes-factor values.
        """

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
        """Dispatch posterior reweighting by model assumption.

        Args:
            random_state (int): Random seed for weighted sampling.

        Returns:
            pd.DataFrame: Reweighted posterior samples.
        """

        if self.assume_parameter_independence:
            return self.get_reweighted_samples_1d(random_state=random_state)

        return self.get_reweighted_samples_dd(random_state=random_state)
