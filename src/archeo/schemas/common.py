from dataclasses import dataclass
from typing import Optional


@dataclass
class Domain:
    """Domain of a parameter."""

    low: Optional[float] = None
    high: Optional[float] = None
