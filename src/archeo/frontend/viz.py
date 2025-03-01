import pandas as pd
import plotly.graph_objects as go


def add_pdf(fig: go.Figure, series: pd.Series, label: str) -> None:
    """Plot the distribution of a column in a dataframe."""

    fig.add_trace(
        go.Histogram(
            x=series,
            histnorm="probability density",
            opacity=0.75,
            name=label,
        )
    )
