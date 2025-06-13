import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Data Visualization Exercise

    With the permission of my teacher, I used this exercise as an opportunity to visualize model performance for my data challenge. The short explanation is that I created a music recommender using embeddings and linUCB for psyrock. The training dataset is a collection on all my music on spotify with 4 categories: 'not' (completley different genre), 'near' (from psyrock artist but songs I don't like i particular), 'liked' (liked psyrock), and 'loved' (subset of liked that is most representational of my taste). It converges nicely to my taste already, but I want a deeper look into the exploration behavior.

    **Quick note**: this notebook was made in Marimo, allowing for more interactivity. If you want the full experience you should run marimo using this notebook in the [project repository](https://github.com/RobertvdLeeuw/suggest), that also contains a 100Mb example of the full (1.4Gb) dataset.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Setup

    First things first, embeddings have to be loaded and reduced, and the model has to be ran.
    """
    )
    return


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
    model = LinUCB(n_dim=250, alpha=3, feature_start=-1)

    from metrics.general import GoodSongsLeft, SongsPicked
    from metrics.linear import SongValuesUCB, CoefficientChange

    t = model.train(reduced, metrics=[SongValuesUCB, 
                                      GoodSongsLeft, 
                                      CoefficientChange,
                                      SongsPicked], 
                    T=1000)
    return CoefficientChange, GoodSongsLeft, SongValuesUCB, SongsPicked, t


@app.cell(hide_code=True)
def _(CoefficientChange, GoodSongsLeft, SongValuesUCB, mo, t):
    from plots import plot_all
    fig2 = plot_all([t], [GoodSongsLeft(), 
                          CoefficientChange(), 
                          SongValuesUCB(agg_how="mean")])
    # fig2.add_vline(x=timestep.value)

    mo.ui.plotly(
        fig2
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""These are my previous visuals. As stated previously, the model is indeed recommending a lot of the good songs in the dataset, but these don't help me understand the exploratory side all too much.""")
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
    mo.md(
        r"""
    # Timestep Based Visuals
    To get a better understanding of the exploration, I designed the following plots.
    The first is a UMAP reduction of my space, showing only only songs recommended before or at timestep $t$. This allows me to see where in the space the model is picking from, which is particularly useful in early timsteps.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    timestep = mo.ui.slider(start=1, stop=999, step=1, value=1, label="Timestep")
    classes = mo.ui.multiselect(options=["not", "near", "liked", "loved"],
                                value=["not", "near", "liked", "loved"])


    timestep, classes
    return classes, timestep


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
    return np, value_over_time


@app.cell(hide_code=True)
def _(SongsPicked, classes, cont, mo, px, reduced_umap, t, timestep, trust):
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
                         subtitle=f"Reduction trustworthiness: {trust:.3f}, continuity: {cont:.3f}",
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
def _(mo):
    mo.md(r"""Below is a histogram of predicted values of *all* songs at timestep $t$. I believe most value comes from this plot when inspecting the overlap of not/near into liked/loved. Without inspecting individual picks, I would presume 'exploratory moves' are sourced from there, and I should be able to influence this overlap with different hyperparams and reward functions. Note that in the earliest timesteps the values are all over the place and they need a couple steps to fall within the usual range.""")
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
    return (px,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Data Enrichment: Genres
    These plots and the songs in them make sense to *me* because it's my music, but to get across the (exploring) performance more I need genres to classify by. I enriched my dataset with genres from the MusicBrainz API. Unfortunely, these are community driven, meaning low coherence and coverage. About 40% of my embedded songs are tagged, but that should be enough to get *some* insights. Plus, I can easily swap out the genre datasource later down the line. 
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Preparation
    Below are the counts of genre's we will consider (there were many more with <9 occurences). 'Rock' will be removed because it's too wide.vague of a descriptor considering the other niches listed. This is still too many genres for meaningful comparison, so I created a manual aggregation of these genres. 

    Another quirk in the data is that songs usually are tagged with multiple genres. I didn't want to simply pick one from these, so any score or picked status tied to a songs will count as 1 occurence for each of its genres. For the short term this seems like a better representation.
    """
    )
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

    mo.ui.plotly(px.bar(genre_counts, 
                        color=genre_counts.index,
                        title="Aggregated genre counts"))
    return (genre_mappings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Analysis

    Now, with the same timestep, we can analyze overall genre visuals during training.
    The second plot is more detailed (histograms instead of boxplots), which would still to unreadable with this amount of genres, hence the dropdown selection. (Note: the second plot is with matplotlib because the plotly implementation was too slow).
    """
    )
    return


@app.cell(hide_code=True)
def _(genres_tags, mo, pd, px, timestep, value_over_time):
    song_to_genres = {k: list(set(v)) for k, v in genres_tags.groupby('name')['genre'].apply(list).to_dict().items()}

    # Expand the original dataframe
    expanded_rows = []
    for h, row in value_over_time[value_over_time.t == timestep.value].iterrows():
        song = row['song']

        for genre in song_to_genres.get(song, []):
            new_row = row.copy()
            new_row['genre'] = genre
            expanded_rows.append(new_row)

    mo.ui.plotly(
        px.box(pd.DataFrame(expanded_rows)[["value", "genre"]], y="value", x="genre", color="genre")
    )
    return (song_to_genres,)


@app.cell
def _(genre_mappings, mo, timestep):
    genres = mo.ui.multiselect(options=set(genre_mappings.values()),
                                value=[])
    timestep, genres
    return (genres,)


@app.cell(hide_code=True)
def _(SongsPicked, genres, np, pd, song_to_genres, t):
    if genres.value:
        t_genres = [song_to_genres.get(song.split('- ')[-1].replace(".wav.csv", ""), ["Other"])
        for song in t.metrics[SongsPicked.name].values]

        from sklearn.preprocessing import MultiLabelBinarizer

        # Create the binary matrix
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(t_genres)

        import matplotlib.pyplot as plt

        # Create DataFrame with proper column names
        genre_df = pd.DataFrame(
            genre_matrix,
            columns=mlb.classes_,
            index=range(len(t_genres))
        )

        genre_df.reset_index(inplace=True)
        genre_df.rename(columns={'index': 't'}, inplace=True)

        from plots import moving_average
        def plotthis():
            all_genres = genres.value

            # Create binary matrix
            binary_data = []
            for t, genres_at_t in enumerate(t_genres):
                row = {'t': t}
                for genre in all_genres:
                    row[genre] = 1.0 if genre in genres_at_t else 0.0
                binary_data.append(row)

            genre_df = pd.DataFrame(binary_data).astype(float)


            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 8))

            # Method 1: Fully overlaid bars
            mov_avg = 25
            x = pd.Series(genre_df['t'])[:-(mov_avg-1)]
            width = 0.8  # Width of bars

            colors = plt.cm.Set3(np.linspace(0, 1, len(all_genres)))  # Generate distinct colors

            for i, genre in enumerate(reversed(all_genres)):
                # print(genre_df[genre])
                y = moving_average(list(genre_df[genre]), n=mov_avg)
                ax.bar(x, y, width=width, alpha=0.7, label=genre, color=colors[i])

            ax.set_xlabel('Timestep (t)', fontsize=12)
            ax.set_ylabel('Genre Present (1=Yes, 0=No)', fontsize=12)
            ax.set_title('Genre Presence Over Time', fontsize=14, fontweight='bold')
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 1])
            ax.set_xticks(range(len(t_genres)))
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.show()
        plotthis()
    else:
        print("Pick genre(s) from the dropdown above.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This plot would be better if the genres were equally distributed, but that's not worth the effort with the lowe coverage of the genres. That said, it's still great to be able to see genre exploration over time!""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Conclusion: 
    I'm happy with how these plots ended up, and I already started using them for model performance exploration - proving to me I've visualized actually useful info. The code in this notebook isn't the cleanest but I'm in the middle of rewriting the metric tracking system anyway so all these plot will be redone soon.
    """
    )
    return


if __name__ == "__main__":
    app.run()
