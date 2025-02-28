import pandas as pd
import plotly.graph_objects as go


def plot_pdf(df: pd.DataFrame, col: str, label: str) -> go.Figure:
    """Plot the distribution of a column in a dataframe."""

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df[col],
            histnorm="density",
            marker=dict(color="blue", line=dict(width=2)),
            opacity=1,
        )
    )

    fig.update_layout(
        title="Probability Density Function",
        xaxis_title=label,
        yaxis_title="Density",
        bargap=0.05,
        showlegend=False,
    )

    return fig
