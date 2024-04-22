import enum
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Padding:
    """
    Padding for plot.

    Attributes
    ----
    tpad: float, the top padding of the plot.
    lpad: float, the left padding of the plot.
    bpad: float, the bottom padding of the plot.
    """

    tpad: float = 2.5
    lpad: float = 0.1
    bpad: float = 0.12


@dataclass
class Labels:
    """
    Labels for plot.

    Attributes
    ---
    title: str, the title of the plot.
    xlabel: str, the x-axis label of the plot.
    ylabel: str, the y-axis label of the plot.
    zlabel: str, the z-axis label of the plot.
    """

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""


class Color(enum.Enum):
    """Color for plot."""

    BLUE = "blue"
    ORANGE = "orange"
    GREY = "grey"
    GREEN = "green"
    RED = "red"
    PINK = "pink"
    PURPLE = "purple"
    YELLOW = "yellow"
    BROWN = "brown"
    CYAN = "cyan"

    @staticmethod
    def value_iter() -> Iterator[str]:
        """Get the generator of the color."""

        return iter([color.value for color in Color])
