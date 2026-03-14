from abc import ABC, abstractmethod
from typing import Optional, TypeAlias, Union

import numpy as np
from pydantic import BaseModel, PositiveFloat, field_validator


Weights: TypeAlias = PositiveFloat


class DistributionBase(ABC):
    """Distribution of a parameter."""

    @property
    @abstractmethod
    def min(self) -> float:
        """Minimum value of the distribution."""

    @property
    @abstractmethod
    def max(self) -> float:
        """Maximum value of the distribution."""

    @abstractmethod
    def draw(self, size: Optional[int] = None) -> Union[float, np.ndarray[float]]:
        """Draw a random value from the distribution."""


class Uniform(BaseModel, DistributionBase, frozen=True):
    """Uniform distribution.

    Attributes:
        low (float): Lower bound of the distribution
        high (float): Upper bound of the distribution
    """

    low: float = float("-inf")
    high: float = float("inf")

    @property
    def min(self) -> float:
        """Minimum value of the distribution."""

        return self.low

    @property
    def max(self) -> float:
        """Maximum value of the distribution."""

        return self.high

    def draw(self, size: Optional[int] = None) -> Union[float, np.ndarray[float]]:
        """Draw a random value from the domain (uniform)."""

        return np.random.uniform(low=self.low, high=self.high, size=size)


class Normal(BaseModel, DistributionBase, frozen=True):
    """Normal distribution.

    Attributes:
        mean (float): Mean of the distribution
        std (float): Standard deviation of the distribution
    """

    mean: float = 0.0
    std: float = 1.0

    @property
    def min(self) -> float:
        """Minimum value of the distribution."""

        return float("-inf")

    @property
    def max(self) -> float:
        """Maximum value of the distribution."""

        return float("inf")

    def draw(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw a random value from the domain (normal)."""

        return np.random.normal(loc=self.mean, scale=self.std, size=size)


class PiecewiseUniform(BaseModel, DistributionBase, frozen=True):
    """Piecewise uniform distribution.

    Attributes:
        uniforms (dict[Uniform, Weights]): A dictionary of uniform distributions and their corresponding weights.
    """

    uniforms: dict[Uniform, Weights] = {}

    @field_validator("uniforms", mode="before")
    @classmethod
    def validate_total_weights(cls, v):
        """Validate that the total weights sum to 1."""

        total_weights = sum(v.values())
        if total_weights != 1.0:
            raise ValueError(f"Total weights must sum to 1. Currently: {total_weights}")
        return v

    @property
    def min(self) -> float:
        """Minimum value of the distribution."""

        return min(uniform.low for uniform in self.uniforms)

    @property
    def max(self) -> float:
        """Maximum value of the distribution."""

        return max(uniform.high for uniform in self.uniforms)

    def _draw_multiple(self, size: int) -> np.ndarray:
        """Draw multiple random values from the domain (piecewise uniform)."""

        sizes = {uniform: int(size * weights) for uniform, weights in self.uniforms.items()}
        sample_chunks = [uniform.draw(size=sizes[uniform]) for uniform in self.uniforms]
        remaining = size - sum(sizes.values())
        if remaining > 0:
            sample_chunks.append([self.draw() for _ in range(remaining)])

        samples = np.concatenate(sample_chunks)
        np.random.shuffle(samples)
        return samples

    def draw(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw a random value from the domain (piecewise uniform)."""

        if size and (size > 1):
            return self._draw_multiple(size)

        # Select a uniform distribution based on weights
        selected_uniform = np.random.choice(list(self.uniforms.keys()), p=list(self.uniforms.values()))
        return selected_uniform.draw()
