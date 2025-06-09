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

    reduced = pca(EMBEDDINGS, n_dim=250)

    model = LinUCB(n_dim=250, alpha=1, feature_start=-1)

    from metrics.general import GoodSongsLeft
    from metrics.linear import SongValuesUCB

    t = model.train(reduced, metrics=[SongValuesUCB, 
                                      GoodSongsLeft], T=1000)

    t
    return GoodSongsLeft, LinUCB, SongValuesUCB, mo, reduced, t


@app.cell
def _(GoodSongsLeft, LinUCB, SongValuesUCB, reduced):
    t2 = LinUCB(n_dim=250, alpha=3, feature_start=-1).train(reduced, metrics=[SongValuesUCB, 
                                                   GoodSongsLeft], T=1000)
    return (t2,)


@app.cell
def _(GoodSongsLeft, SongValuesUCB, mo, t, t2):
    from plots import plot_all

    mo.ui.plotly(
        plot_all([t, t2], [GoodSongsLeft(), 
                       SongValuesUCB(agg_how="max"), 
                       SongValuesUCB(agg_how="mean")])
    )
    return


@app.cell
def _(t):
    len([t])
    return


if __name__ == "__main__":
    app.run()
