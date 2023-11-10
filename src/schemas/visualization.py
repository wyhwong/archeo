import dataclasses


@dataclasses.dataclass
class Padding:
    """
    Padding for a plot.

    Attributes
    ----------
    tpad: float
        the top padding of the plot.
    lpad: float
        the left padding of the plot.
    bpad: float
        the bottom padding of the plot.
    """

    tpad: float = 2.5
    lpad: float = 0.1
    bpad: float = 0.12


@dataclasses.dataclass
class Labels:
    """
    Labels for a plot.

    Attributes
    ----------
    title: str
        the title of the plot.
    xlabel: str
        the label of the x-axis.
    ylabel: str
        the label of the y-axis.
    """

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
