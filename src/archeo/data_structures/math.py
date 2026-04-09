from abc import ABC, abstractmethod
from typing import Union, overload

import numpy as np
from pydantic import BaseModel


class DomainBase(BaseModel, ABC):
    """Base class for domains of parameters."""

    @overload
    def contains(self, value: float) -> bool: ...

    @overload
    def contains(self, value: np.ndarray) -> np.ndarray: ...

    @overload
    def not_contains(self, value: float) -> bool: ...

    @overload
    def not_contains(self, value: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def contains(self, value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check if a value is within the domain."""

    @abstractmethod
    def not_contains(self, value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check if a value is not within the domain."""


class Domain(BaseModel, frozen=True):
    """Domain of a parameter."""

    low: float = float("-inf")
    high: float = float("inf")

    def contains(self, value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check if a value is within the domain."""

        if isinstance(value, np.ndarray):
            return (value <= self.high) & (value >= self.low)

        return self.low <= value <= self.high

    def not_contains(self, value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check if a value is not within the domain."""

        if isinstance(value, np.ndarray):
            return (value > self.high) | (value < self.low)

        return not self.contains(value)

    def to_tuple(self) -> tuple[float, float]:
        """Convert the domain to a tuple."""

        return self.low, self.high


class PiecewiseDomain(BaseModel, frozen=True):
    """Piecewise domain of a parameter."""

    domains: list[Domain] = []

    def contains(self, value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check if a value is within the piecewise domain."""

        if isinstance(value, np.ndarray):
            return np.any([domain.contains(value) for domain in self.domains], axis=0)

        return any(domain.contains(value) for domain in self.domains)

    def not_contains(self, value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check if a value is not within the piecewise domain."""

        if isinstance(value, np.ndarray):
            return np.all([domain.not_contains(value) for domain in self.domains], axis=0)

        return all(domain.not_contains(value) for domain in self.domains)
