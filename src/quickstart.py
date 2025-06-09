import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from models.linear import LinUCB
    from reduce import pca

    import pandas as pd

    EMBEDDINGS = pd.read_csv("../data/enriched.csv")

    reduced = pca(EMBEDDINGS, n_dim=250)

    model = LinUCB(n_dim=250, alpha=1, feature_start=-1)

    from metrics.general import GoodSongsLeft
    from metrics.linear import SongValuesUCB

    t = model.train(reduced, metrics=[SongValuesUCB, GoodSongsLeft], T=1000)

    from plots import plot_all

    t
    #mo.ui.plotly(
    #    plot_all([t], [SongValuesUCB, GoodSongsLeft])
    #)
    return


if __name__ == "__main__":
    app.run()
