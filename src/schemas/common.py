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

    def in_bound(self, value: float) -> bool:
        """
        Check if a value is in the domain.

        Parameters
        ----------
        value : float
            Value to be checked.

        Returns
        -------
        in_bound : bool
            Whether the value is in the domain.
        """
        return self.low <= value <= self.high
