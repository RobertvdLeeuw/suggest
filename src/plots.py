from objects import Metric, Trajectory, METADATA_COLS

from metrics.reduce import trust_cont

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


COLORS = ["#1f77b4", "#ff7f0e", "#5b3491", "#37c42d"]
LINES = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]

# Use dash for stuff like legend/slicers
def plot_all(trajectories: list[Trajectory], metrics: list[Metric] = [], moving_avg: int = 15) -> go.Figure:
    if not isinstance(trajectories, list):
        trajectories = [trajectories]
    if not isinstance(metrics, list):
        metrics = [metrics]

    missing = [f"{t.model_name}: {m.name}" for m in metrics for t in trajectories if m.name not in t.metrics]
    assert missing == [], f"Required metric(s) missing for some trajectories:\n\t{'\n\t'.join(missing)}"

    assert len(trajectories) < 4, "That'll look like an epileptic attack, retard. Back to the drawing board with you!"

    fig = make_subplots(
        rows=len(metrics), 
        cols=1,
        subplot_titles=[m.name for m in metrics],
        vertical_spacing=0.1,
        shared_xaxes=True,
        # specs=[[{"secondary_y": False} for _ in range(len(histories))] for _ in range(2)]
    )
    fig['layout'].update(height=300*len(metrics))

    for i, m in enumerate(metrics, start=1):
        for j, (t, col) in enumerate(zip(trajectories, COLORS)):
            for line_style, (component_name, component_values) in zip(LINES, t.metrics[m.name].items()):
                if isinstance(component_values[0], dict):
                    component_values = [list(values_at_t.values()) for values_at_t in component_values]
                    match m.agg_how:
                        case "mean":
                            component_values = np.mean(component_values, axis=1)

                        case "max":
                            component_values = np.max(component_values, axis=1)

                        case _:
                            raise NotImplementedError(f"Unknown aggregation method: '{m.agg_how}'")

                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(component_values) - moving_avg + 1)),
                        y=moving_average(component_values, moving_avg),
                        name=component_name,
                        legendgroup=t.model_name,
                        showlegend= i == 1,  # Only when multiple components and on first model.
                        line={"width": 2, "color": col, "dash": line_style}
                    ),
                    row=i, col=1
                )
    fig.update_xaxes(title_text="Steps", row=len(metrics), col=1)
    fig.update_layout(
        title_text=f'Model Performance (moving avg of {moving_avg})',
        showlegend=True
    )

    return fig

def filter_sparse_unliked(data: pd.DataFrame, frac: float) -> pd.DataFrame:
    return pd.concat([data[data.score == "not"].sample(frac=frac), 
                      data[data.score == "near"].sample(frac=frac),
                      data[data.score.isin(["liked", "loved"])]]).sort_values(by="score")


def scatter_3d(embeddings: pd.DataFrame, reduced: pd.DataFrame, 
               trust: float, cont: float,
               filter_rate: float = 0.3, opacity: float = 0.8):
    assert 0 <= filter_rate <= 1
    assert 0 <= opacity <= 1

    # trust, cont = trust_cont(embeddings, reduced)
    reduced_sparse_umap = filter_sparse_unliked(reduced, filter_rate)
    
    reduced_sparse_umap["size"] = 10
    score_sequence = {"not": 1, "near": 2, "liked": 3, "loved": 4}
    fig = px.scatter_3d(data_frame=reduced_sparse_umap.sort_values(by="score", 
                                                                   key=lambda col: col.apply(lambda x: score_sequence[x])),
                        x="x", y="y", z="z", 
                        color="score",
                        color_discrete_map={'not':'red',
                                            'near':'orange',
                                            'liked':'green',
                                            'loved':'darkgreen'},
                        title=f"Reduced Embeddings",
                        subtitle=f"Trustworthiness: {trust:.3f}, Continuity: {cont:.3f}",
                        hover_data={"x": False, 
                                    "y": False, 
                                    "z": False,  
                                    "size": False,  
                                    "name": True,
                                    "chunk": True},
                        size="size",
                        size_max=10,
                        opacity=opacity,
                        # width=800,
                        # height=700
                        )

    for l, _ in enumerate(fig.data):
        fig.data[l].marker.line.width = 0

    return fig
