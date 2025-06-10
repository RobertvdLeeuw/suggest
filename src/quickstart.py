import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from models.linear import LinUCB
    from reduce import pca
    from functools import partial

    import pandas as pd

    EMBEDDINGS = pd.read_csv("../data/enriched.csv")
    EMBEDDINGS = EMBEDDINGS.drop('Unnamed: 0', axis=1)  # Old index

    reduced = pca(EMBEDDINGS, n_dim=250)
    return EMBEDDINGS, LinUCB, mo, pd, reduced


@app.cell
def _(EMBEDDINGS, mo):
    from plots import scatter_3d
    from reduce import umap

    mo.ui.plotly(scatter_3d(EMBEDDINGS, 
                            umap(EMBEDDINGS, 3, spread=5, min_dist=5, n_neighbors=30, densmap=False),
                            0.3, 0.7))
    return


@app.cell
def _(LinUCB, reduced):
    model = LinUCB(n_dim=250, alpha=1, feature_start=-1)

    from metrics.general import GoodSongsLeft
    from metrics.linear import SongValuesUCB

    t = model.train(reduced, metrics=[SongValuesUCB, 
                                      GoodSongsLeft], T=1000)

    t
    return GoodSongsLeft, SongValuesUCB, model, t


@app.cell
def _(GoodSongsLeft, LinUCB, SongValuesUCB, reduced):
    t2 = LinUCB(n_dim=250, alpha=3, feature_start=-1).train(reduced, metrics=[SongValuesUCB, 
                                                   GoodSongsLeft], T=1000)
    return (t2,)


@app.cell(hide_code=True)
def _(GoodSongsLeft, SongValuesUCB, mo, t, t2):
    from plots import plot_all

    mo.ui.plotly(
        plot_all([t, t2], [GoodSongsLeft(), 
                           SongValuesUCB(agg_how="max"), 
                           SongValuesUCB(agg_how="mean")])
    )
    return


@app.cell(hide_code=True)
def _(model, reduced):
    values = model._calc_song_values(reduced, 
                                     reduced[reduced.score.isin(["near", "not"])]["name"].unique())

    for i in range(50):
        best = max(values, key=values.get)
        print(f"\t{i}  {values.pop(best):.3f} - {best.replace('.wav.csv', '')}")
    return


@app.cell(hide_code=True)
def _(mo):
    timestep = mo.ui.slider(start=0, stop=999, step=1, value=0, label="Timestep")
    timestep
    return (timestep,)


@app.cell(hide_code=True)
def _(SongValuesUCB, pd, reduced, t):
    import numpy as np

    value_over_time = pd.DataFrame({
        "song": list(reduced.song.unique()) * t.T,
        "value": [t.metrics[SongValuesUCB.name].values["Total"][step][song]
                  for step in range(t.T) for song in reduced.song.unique()],
        "label": [list(reduced[reduced.song == song].score)[0]
                  for song in reduced.song.unique()] * t.T,
        "t": [step for step in range(t.T) for _ in range(len(reduced.song.unique()))]
    })

    # value_over_time[(value_over_time.t == timestep.value)].value.describe()
    # value_over_time.head()
    return (value_over_time,)


@app.cell(hide_code=True)
def _(mo, timestep, value_over_time):
    import plotly.graph_objects as go
    import plotly.express as px


    fig = go.Figure()
    current = value_over_time[value_over_time.t == timestep.value]
    bins = dict(start=-3, end=3, size=0.025)
    fig.add_trace(go.Histogram(x=current[current.label == "not"]["value"], 
                               xbins=bins,
                               marker_color='#d92727',
                               name="Not"))
    fig.add_trace(go.Histogram(x=current[current.label == "near"]["value"], 
                               xbins=bins,
                               marker_color='#ffe921',
                               name="Near"))
    fig.add_trace(go.Histogram(x=current[current.label == "liked"]["value"], 
                               xbins=bins,
                               marker_color='#51e324',
                               name="Liked"))
    fig.add_trace(go.Histogram(x=current[current.label == "loved"]["value"], 
                               xbins=bins,
                               marker_color='#309611',
                               name="Loved"))
    # fig.add_trace(go.Histogram(x=x1))

    # Overlay both histograms
    fig.update_layout(barmode='overlay', title="Predicted Song Values during Trajectory")
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    fig.update_xaxes(range=[-1.5, 1.5])

    mo.ui.plotly(
        fig
    )
    return


if __name__ == "__main__":
    app.run()
