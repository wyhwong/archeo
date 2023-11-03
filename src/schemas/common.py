import dataclasses


@dataclasses.dataclass
class Domain:
    """
    Domain of a parameter.

    Parameters
    ----------
    low : float
        Lower bound of the domain.
    high : float
        Upper bound of the domain.
    """

    low: float
    high: float
