from abc import ABC, abstractmethod

from pydantic import BaseModel


class DomainBase(BaseModel, ABC):
    """Base class for domains of parameters."""

    @abstractmethod
    def contains(self, value: float) -> bool:
        """Check if a value is within the domain."""


class Domain(BaseModel, frozen=True):
    """Domain of a parameter."""

    low: float = float("-inf")
    high: float = float("inf")

    def contains(self, value: float) -> bool:
        """Check if a value is within the domain."""

        return self.low <= value <= self.high

    def to_tuple(self) -> tuple[float, float]:
        """Convert the domain to a tuple."""

        return self.low, self.high


class PiecewiseDomain(BaseModel, frozen=True):
    """Piecewise domain of a parameter."""

    domains: list[Domain] = []

    def contains(self, value: float) -> bool:
        """Check if a value is within the piecewise domain."""

        return any(domain.contains(value) for domain in self.domains)
