from pydantic import BaseModel


class Padding(BaseModel, frozen=True):
    """Plot label bundle for title and axis annotations."""

    tpad: float = 2.5
    lpad: float = 0.1
    bpad: float = 0.12


class Labels(BaseModel, frozen=True):
    """Figure layout padding configuration used by plotting helpers."""

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
