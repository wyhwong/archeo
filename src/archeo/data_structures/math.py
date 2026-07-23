from abc import ABC, abstractmethod
from typing import Union, overload

import numpy as np
from pydantic import BaseModel, Field


class DomainBase(BaseModel, ABC):
    """Abstract domain interface for scalar/array membership checks.

    Implementations define `contains` and `not_contains` for validating whether
    values lie inside or outside allowed numerical regions.
    """

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
        """Return whether value(s) lie within the domain.

        Args:
            value (Union[float, np.ndarray]): Scalar or array to test.

        Returns:
            Union[bool, np.ndarray]: Inclusion mask or scalar flag.
        """

    @abstractmethod
    def not_contains(self, value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Return whether value(s) lie outside the domain.

        Args:
            value (Union[float, np.ndarray]): Scalar or array to test.

        Returns:
            Union[bool, np.ndarray]: Exclusion mask or scalar flag.
        """


class Domain(BaseModel, frozen=True):
    """Closed 1D interval domain `[low, high]`.

    Supports vectorized membership tests and conversion to tuple form for
    downstream validation and display.
    """

    low: float = float("-inf")
    high: float = float("inf")

    def contains(self, value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check membership in a closed interval `[low, high]`.

        Args:
            value (Union[float, np.ndarray]): Scalar or array to test.

        Returns:
            Union[bool, np.ndarray]: Inclusion result.
        """

        if isinstance(value, np.ndarray):
            return (value <= self.high) & (value >= self.low)

        return self.low <= value <= self.high

    def not_contains(self, value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check non-membership in a closed interval `[low, high]`.

        Args:
            value (Union[float, np.ndarray]): Scalar or array to test.

        Returns:
            Union[bool, np.ndarray]: Exclusion result.
        """

        if isinstance(value, np.ndarray):
            return (value > self.high) | (value < self.low)

        return not self.contains(value)

    def to_tuple(self) -> tuple[float, float]:
        """Return interval bounds as a tuple.

        Returns:
            tuple[float, float]: `(low, high)`.
        """

        return self.low, self.high


class PiecewiseDomain(BaseModel, frozen=True):
    """Union of multiple 1D domains.

    A value belongs to this domain if it belongs to at least one constituent
    subdomain.
    """

    domains: list[Domain] = Field(default_factory=list)

    def contains(self, value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check membership in a union of intervals.

        Args:
            value (Union[float, np.ndarray]): Scalar or array to test.

        Returns:
            Union[bool, np.ndarray]: Inclusion result across all subdomains.
        """

        if isinstance(value, np.ndarray):
            return np.any([domain.contains(value) for domain in self.domains], axis=0)

        return any(domain.contains(value) for domain in self.domains)

    def not_contains(self, value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check non-membership in a union of intervals.

        Args:
            value (Union[float, np.ndarray]): Scalar or array to test.

        Returns:
            Union[bool, np.ndarray]: Exclusion result across all subdomains.
        """

        if isinstance(value, np.ndarray):
            return np.all([domain.not_contains(value) for domain in self.domains], axis=0)

        return all(domain.not_contains(value) for domain in self.domains)
