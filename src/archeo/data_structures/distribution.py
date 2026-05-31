from abc import ABC, abstractmethod
from typing import Optional, TypeAlias, Union

import numpy as np
from pydantic import BaseModel, PositiveFloat, field_validator


Weights: TypeAlias = PositiveFloat


class DistributionBase(ABC):
    """Abstract probability-distribution interface.

    Concrete implementations expose support bounds (`min`, `max`) and a `draw`
    method for scalar or vectorized sampling.
    """

    @property
    @abstractmethod
    def min(self) -> float:
        """Return lower bound of distribution support.

        Returns:
            float: Minimum supported value.
        """

    @property
    @abstractmethod
    def max(self) -> float:
        """Return upper bound of distribution support.

        Returns:
            float: Maximum supported value.
        """

    @abstractmethod
    def draw(self, size: Optional[int] = None) -> Union[float, np.ndarray[float]]:
        """Draw sample(s) from the distribution.

        Args:
            size (Optional[int]): Number of draws. `None` requests a scalar draw.

        Returns:
            Union[float, np.ndarray[float]]: Drawn sample(s).
        """


class Uniform(BaseModel, DistributionBase, frozen=True):
    """Uniform distribution over a finite interval `[low, high]`."""

    low: float = float("-inf")
    high: float = float("inf")

    @property
    def min(self) -> float:
        """Return lower bound of uniform support.

        Returns:
            float: Lower bound.
        """

        return self.low

    @property
    def max(self) -> float:
        """Return upper bound of uniform support.

        Returns:
            float: Upper bound.
        """

        return self.high

    def draw(self, size: Optional[int] = None) -> Union[float, np.ndarray[float]]:
        """Draw sample(s) from a uniform distribution.

        Args:
            size (Optional[int]): Number of draws.

        Returns:
            Union[float, np.ndarray[float]]: Drawn sample(s).
        """

        return np.random.uniform(low=self.low, high=self.high, size=size)


class Normal(BaseModel, DistributionBase, frozen=True):
    """Gaussian distribution parameterized by mean and standard deviation."""

    mean: float = 0.0
    std: float = 1.0

    @property
    def min(self) -> float:
        """Return lower support bound for a normal distribution.

        Returns:
            float: Negative infinity.
        """

        return float("-inf")

    @property
    def max(self) -> float:
        """Return upper support bound for a normal distribution.

        Returns:
            float: Positive infinity.
        """

        return float("inf")

    def draw(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw sample(s) from a normal distribution.

        Args:
            size (Optional[int]): Number of draws.

        Returns:
            Union[float, np.ndarray]: Drawn sample(s).
        """

        return np.random.normal(loc=self.mean, scale=self.std, size=size)


class PiecewiseUniform(BaseModel, DistributionBase, frozen=True):
    """Weighted mixture of uniform segments.

    Represents a piecewise-uniform density defined by interval-weight pairs whose
    weights sum to 1.
    """

    uniforms: dict[Uniform, Weights] = {}

    @field_validator("uniforms", mode="before")
    @classmethod
    def validate_total_weights(cls, v):
        """Validate piecewise weights sum to 1.

        Args:
            v: Mapping from Uniform segment to weight.

        Returns:
            Any: Validated mapping.

        Raises:
            ValueError: If total weight differs from 1.
        """

        total_weights = sum(v.values())
        if total_weights != 1.0:
            raise ValueError(f"Total weights must sum to 1. Currently: {total_weights}")
        return v

    @property
    def min(self) -> float:
        """Return minimum lower bound among piecewise segments.

        Returns:
            float: Global lower bound.
        """

        return min(uniform.low for uniform in self.uniforms)

    @property
    def max(self) -> float:
        """Return maximum upper bound among piecewise segments.

        Returns:
            float: Global upper bound.
        """

        return max(uniform.high for uniform in self.uniforms)

    def _draw_multiple(self, size: int) -> np.ndarray:
        """Draw multiple samples according to piecewise segment weights.

        Args:
            size (int): Number of draws.

        Returns:
            np.ndarray: Shuffled concatenated samples.
        """

        sizes = {uniform: int(size * weights) for uniform, weights in self.uniforms.items()}
        sample_chunks = [uniform.draw(size=sizes[uniform]) for uniform in self.uniforms]
        remaining = size - sum(sizes.values())
        if remaining > 0:
            sample_chunks.append([self.draw() for _ in range(remaining)])

        samples = np.concatenate(sample_chunks)
        np.random.shuffle(samples)
        return samples

    def draw(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw sample(s) from a piecewise-uniform distribution.

        Args:
            size (Optional[int]): Number of draws. If omitted or <=1, draws one sample.

        Returns:
            Union[float, np.ndarray]: Drawn sample(s).
        """

        if size and (size > 1):
            return self._draw_multiple(size)

        # Select a uniform distribution based on weights
        selected_uniform = np.random.choice(list(self.uniforms.keys()), p=list(self.uniforms.values()))
        return selected_uniform.draw()
