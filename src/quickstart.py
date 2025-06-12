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


@app.cell(hide_code=True)
def _(LinUCB, reduced):
    model = LinUCB(n_dim=250, alpha=1, feature_start=-1)

    from metrics.general import GoodSongsLeft, SongsPicked
    from metrics.linear import SongValuesUCB, CoefficientChange

    t = model.train(reduced, metrics=[SongValuesUCB, 
                                      GoodSongsLeft, 
                                      CoefficientChange,
                                      SongsPicked], 
                    T=1000)
    return (
        CoefficientChange,
        GoodSongsLeft,
        SongValuesUCB,
        SongsPicked,
        model,
        t,
    )


@app.cell(hide_code=True)
def _(CoefficientChange, GoodSongsLeft, LinUCB, SongValuesUCB, reduced):
    t2 = LinUCB(n_dim=250, alpha=3, feature_start=-1).train(reduced, 
                                                            metrics=[SongValuesUCB, 
                                                                     GoodSongsLeft, 
                                                                     CoefficientChange], 
                                                            T=1000)
    return (t2,)


@app.cell(hide_code=True)
def _(CoefficientChange, GoodSongsLeft, SongValuesUCB, mo, t, t2):
    from plots import plot_all
    fig2 = plot_all([t, t2], [GoodSongsLeft(), 
                              CoefficientChange(), 
                              SongValuesUCB(agg_how="mean")])
    # fig2.add_vline(x=timestep.value)

    mo.ui.plotly(
        fig2
    )
    return


@app.cell(hide_code=True)
def _(EMBEDDINGS):
    from metrics.reduce import trust_cont
    from reduce import umap


    reduced_umap = umap(EMBEDDINGS, 3, 
                        spread=5, min_dist=5, n_neighbors=30, 
                        densmap=False)

    trust, cont = trust_cont(EMBEDDINGS, reduced_umap)
    return cont, reduced_umap, trust


@app.cell(hide_code=True)
def _(mo):
    unliked_filter_rate = mo.ui.slider(start=0, stop=0.95, step=0.05, value=0.7, label="Unliked filter rate")
    unliked_filter_rate
    return (unliked_filter_rate,)


@app.cell(hide_code=True)
def _(mo, px, reduced_umap, unliked_filter_rate):
    #from metrics.reduce import trust_cont
    from plots import filter_sparse_unliked

    reduced_sparse_umap = filter_sparse_unliked(reduced_umap, 1-unliked_filter_rate.value)

    distribution = reduced_sparse_umap.score.value_counts()

    mo.ui.plotly(
      px.pie(values=distribution, 
             names=distribution.index,
             color=distribution.index,
             color_discrete_map={'not':'red',
                                 'near':'orange',
                                 'liked':'green',
                                 'loved':'darkgreen'},
             title='Score Distribution')
    )
    return


@app.cell(hide_code=True)
def _(EMBEDDINGS, cont, mo, reduced_umap, trust, unliked_filter_rate):
    from plots import scatter_3d
    mo.ui.plotly(scatter_3d(EMBEDDINGS, 
                            reduced_umap,
                            trust, cont,
                            filter_rate=1-unliked_filter_rate.value,
                            opacity=0.7))
    return


@app.cell(hide_code=True)
def _(current, mo, px):
    # genres_tags

    mo.ui.plotly(
        px.box(current, y="value", x="label", color="label")
    )
    return


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
def _(SongsPicked, classes, mo, px, reduced_umap, t, timestep):
    reduced_umap["picked"] = reduced_umap.name.apply(
        lambda x: x in t.metrics[SongsPicked.name].values[:timestep.value]
    )
    reduced_umap["size"] = 10
    score_sequence = {"not": 4, "near": 3, "liked": 2, "loved": 1}

    shrunk = reduced_umap[(reduced_umap.picked) & (reduced_umap.score.isin(classes.value))]

    fig3 = px.scatter_3d(data_frame=shrunk.sort_values(by="score", 
                                                       key=lambda col: col.apply(lambda x: score_sequence[x])),
                         x="x", y="y", z="z", 
                         color="score",
                         color_discrete_map={'not':'red',
                                             'near':'orange',
                                             'liked':'green',
                                             'loved':'darkgreen'},
                         title=f"Songs Recommended at Timestep {timestep.value}",
                         #subtitle=f"Trustworthiness: {trust:.3f}, Continuity: {cont:.3f}",
                         hover_data={"x": False, 
                                     "y": False, 
                                     "z": False,  
                                     "size": False,  
                                     "name": True,
                                     "chunk": True},
                         size="size",
                         size_max=10,
                         opacity=0.85,
                         # width=800,
                         # height=700
                         )

    for l, _ in enumerate(fig3.data):
        fig3.data[l].marker.line.width = 0

    mo.ui.plotly(fig3)
    return


@app.cell(hide_code=True)
def _(classes, mo, timestep, value_over_time):
    import plotly.graph_objects as go
    import plotly.express as px


    fig = go.Figure()
    current = value_over_time[(value_over_time.t == timestep.value) & 
                              (value_over_time.label.isin(classes.value))]
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
    fig.update_layout(barmode='overlay', title=f"Predicted Song Values at Timestep {timestep.value}")
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    fig.update_xaxes(range=[-1.5, 1.5])

    mo.ui.plotly(
        fig
    )
    return current, px


@app.cell(hide_code=True)
def _(SongsPicked, mo, pd, reduced, t):
    prop_instead = mo.ui.switch(value=False, label="Show proportional instead")


    pick_distribution_at_t = [reduced[reduced["name"] == song].score.iloc[0] 
                              for song in t.metrics[SongsPicked.name].values]

    def create_pivoted_proportions(scores):
        df = pd.DataFrame({'score': scores, 'timestep': range(1, len(scores) + 1)})

        # Create running proportions
        running_data = []
        for category in ['not', 'near', 'liked', 'loved']:
            running_counts = (df['score'] == category).cumsum()
            running_props = running_counts / df['timestep']

            for t in range(len(scores)):
                running_data.append({
                    'timestep': t + 1,
                    'type': category,
                    'class count': running_counts.iloc[t],
                    'proportional total': running_props.iloc[t]
                })

        result_df = pd.DataFrame(running_data)
        return result_df

    # Usage
    df_pivoted = create_pivoted_proportions(pick_distribution_at_t)


    prop_instead
    return df_pivoted, prop_instead


@app.cell(hide_code=True)
def _(df_pivoted, mo, prop_instead, px):
    from itertools import accumulate

    #df_pivoted
    mo.ui.plotly(
        px.bar(df_pivoted, x="timestep", 
               y="proportional total" if prop_instead.value else "class count", 
               color="type",
               hover_data={"timestep": True, 
                           "proportional total": False, 
                           "class count": True},
               color_discrete_map={'not':'red',
                                   'near':'orange',
                                   'liked':'green',
                                   'loved':'darkgreen'})
    )  # TOTO: Overlay plot of total counts.
    return


@app.cell(hide_code=True)
def _(mo, pd, px):
    import json, re

    with open("../data/spotify_new.json", "r") as f:
        data = json.load(f)

    genres_tags = [{"name": re.sub(r"'|\"|&|#|\[|\]|:|;", "", data[i]["name"]).strip(),
                    "isrc": data[i]["isrc"],
                    "genres": list(set(data[i]["genres"] +
                                       data[i]["tags"]))
                   } for i in range(len(data)) 
                     if "genres" in data[i]]    

    genres_tags = [{"name": entry["name"].split('-')[-1].replace(".wav.csv", "").strip(),
                    "isrc": entry["isrc"],
                    "genre": genre}
                   for entry in genres_tags
                   for genre in entry["genres"]
                   if genre != "rock"]

    genres_tags = pd.DataFrame(genres_tags)
    genre_counts = genres_tags.genre.value_counts()


    filtered_genres = genre_counts[genre_counts > 10]
    mo.ui.plotly(
        px.bar(filtered_genres)
    )

    # TODO: Rock = anyof(psyrock, kraut, etc), jazz = anyof (jazz, lounge, etc)
    # THEN: piechart of custom agg genres,
    # THEN: All other genre-based plots.
    return genre_counts, genres_tags


@app.cell(hide_code=True)
def _(genre_counts, genres_tags, mo, px):
    from collections import defaultdict
    genre_mappings = defaultdict(lambda: "Other", {
        # Classic Rock
        # "rock": "Classic Rock",
        "classic rock": "Classic Rock", 
        "hard rock": "Classic Rock",
        "arena rock": "Classic Rock",
        "rock and roll": "Classic Rock",
        "blues rock": "Classic Rock",
        "blues-rock": "Classic Rock",

        # Psychedelic Rock (kept separate)
        "psychedelic rock": "Psychedelic Rock",
        "psychedelic": "Psychedelic Rock",

        # Progressive/Art Rock
        "progressive rock": "Progressive/Art Rock",
        "art rock": "Progressive/Art Rock",
        "symphonic rock": "Progressive/Art Rock",
        "rock opera": "Progressive/Art Rock",
        "progressive": "Progressive/Art Rock",
        "progressive-rock": "Progressive/Art Rock",

        # Alternative/Indie Rock
        "alternative rock": "Alternative/Indie Rock",
        "indie rock": "Alternative/Indie Rock",
        "post-punk": "Alternative/Indie Rock",
        "punk": "Alternative/Indie Rock",
        "punk rock": "Alternative/Indie Rock",
        "proto-punk": "Alternative/Indie Rock",
        "alternative punk": "Alternative/Indie Rock",

        # Hip Hop
        "hip hop": "Hip Hop",
        "hip-hop": "Hip Hop",
        "gangsta rap": "Hip Hop",
        "east coast hip hop": "Hip Hop",
        "west coast hip hop": "Hip Hop",
        "conscious hip hop": "Hip Hop",
        "hardcore hip hop": "Hip Hop",
        "jazz rap": "Hip Hop",
        "pop rap": "Hip Hop",
        "g-funk": "Hip Hop",
        "alternative hip hop": "Hip Hop",
        "boom bap": "Hip Hop",
        "underground hip hop": "Hip Hop",

        # Electronic/Dance
        "electronic": "Electronic/Dance",
        "ambient": "Electronic/Dance",
        "synth-pop": "Electronic/Dance",
        "synthpop": "Electronic/Dance",
        "synth pop": "Electronic/Dance",
        "new wave": "Electronic/Dance",
        "dance": "Electronic/Dance",
        "downtempo": "Electronic/Dance",
        "electro": "Electronic/Dance",
        "trip hop": "Electronic/Dance",
        "house": "Electronic/Dance",
        "synthwave": "Electronic/Dance",
        "dance-pop": "Electronic/Dance",
        "alternative dance": "Electronic/Dance",
        "indietronica": "Electronic/Dance",
        "trance": "Electronic/Dance",
        "electropop": "Electronic/Dance",
        "dance-rock": "Electronic/Dance",
        "techno": "Electronic/Dance",
        "industrial": "Electronic/Dance",
        "dub": "Electronic/Dance",
        "synth funk": "Electronic/Dance",

        # Blues/Soul/R&B
        "blues": "Blues/Soul/R&B",
        "soul": "Blues/Soul/R&B",
        "r&b": "Blues/Soul/R&B",
        "funk": "Blues/Soul/R&B",
        "british blues": "Blues/Soul/R&B",
        "electric blues": "Blues/Soul/R&B",
        "contemporary r&b": "Blues/Soul/R&B",
        "pop soul": "Blues/Soul/R&B",
        "soul jazz": "Blues/Soul/R&B",
        "smooth soul": "Blues/Soul/R&B",
        "chicago blues": "Blues/Soul/R&B",
        "motown": "Blues/Soul/R&B",
        "jazz-funk": "Blues/Soul/R&B",

        # Jazz
        "jazz": "Jazz",
        "vocal jazz": "Jazz",
        "swing": "Jazz",
        "big band": "Jazz",
        "jazz fusion": "Jazz",

        # Pop
        "pop": "Pop",
        "pop rock": "Pop",
        "pop/rock": "Pop",
        "art pop": "Pop",
        "psychedelic pop": "Pop",
        "indie pop": "Pop",
        "progressive pop": "Pop",
        "soft rock": "Pop",
        "easy listening": "Pop",
        "ballad": "Pop",

        # Heavy/Metal
        "heavy metal": "Heavy/Metal",
        "metal": "Heavy/Metal",
        "thrash metal": "Heavy/Metal",

        # Experimental (smaller category)
        "experimental": "Experimental",
        "experimental rock": "Experimental",
        "avant-garde": "Experimental",
        "krautrock": "Experimental",
        "garage rock": "Experimental",
        "stoner rock": "Experimental",
        "space rock": "Experimental",
        "acid rock": "Experimental",
        "neo-psychedelia": "Experimental"})

    genre_counts.index = genre_counts.index.map(genre_mappings)
    genres_tags.genre = genres_tags.genre.apply(lambda g: genre_mappings[g])

    mo.ui.plotly(px.bar(genre_counts, color=genre_counts.index))
    return


@app.cell
def _(genres_tags, pd, timestep, value_over_time):
    song_to_genres = {k: list(set(v)) for k, v in genres_tags.groupby('name')['genre'].apply(list).to_dict().items()}

    # Expand the original dataframe
    total = len(value_over_time.index)
    expanded_rows = []
    for h, row in value_over_time[value_over_time.t == timestep.value].iterrows():
        print(f"{h/total*100:.1f}%", end="\r")
        song = row['song']

        for genre in song_to_genres.get(song, []):
            new_row = row.copy()
            new_row['genre'] = genre
            expanded_rows.append(new_row)

    # sum(int(row['song'] in song_to_genres) for _, row in value_over_time.iterrows()) / len(value_over_time.index)*100
    tmp = pd.DataFrame(expanded_rows)
    return


app._unparsable_cell(
    r"""
    mo.ui.plotly(
        px.box(tmp[\"value\", \"genre\"]], y=\"value\", x=\"genre\", color=\"genre\")
    )
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    timestep = mo.ui.slider(start=1, stop=999, step=1, value=1, label="Timestep")
    classes = mo.ui.multiselect(options=["not", "near", "liked", "loved"],
                                value=["not", "near", "liked", "loved"])

    timestep, classes
    return classes, timestep


@app.cell(hide_code=True)
def _(model, reduced):
    values = model._calc_song_values(reduced, 
                                     reduced[reduced.score.isin(["near", "not"])]["name"].unique())

    for i in range(100):
        best = max(values, key=values.get)
        print(f"\t{i}  {values.pop(best):.3f} - {best.replace('.wav.csv', '')}")
    return


if __name__ == "__main__":
    app.run()
