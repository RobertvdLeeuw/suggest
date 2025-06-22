import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Suggest Demo
    This notebook contains a model pretrained on half of the available data and a live feedback system.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from sklearn.model_selection import train_test_split

    from models.linear import LinUCB
    from reduce import pca
    from functools import partial

    import pandas as pd

    EMBEDDINGS = pd.read_csv("../data/enriched.csv")
    EMBEDDINGS = EMBEDDINGS.drop('Unnamed: 0', axis=1)  # Old index

    reduced = pca(EMBEDDINGS, n_dim=250)
    pretrain_reduced, live_reduced = train_test_split(reduced, test_size=0.6)
    return LinUCB, live_reduced, mo, pd, pretrain_reduced, reduced


@app.cell(hide_code=True)
def _(pd):
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

    song_to_genres = {k: list(set(v)) for k, v in genres_tags.groupby('name')['genre'].apply(list).to_dict().items()}
    return (song_to_genres,)


@app.cell(hide_code=True)
def _(LinUCB, live_reduced, pretrain_reduced, song_to_genres):
    from plots import plot_all
    from random import choices
    from objects import Trajectory

    model = LinUCB(n_dim=250, 
                   alpha=0.1, 
                   update_step=5, 
                   feature_start=-1)
    model.train(pretrain_reduced, metrics=[], T=1000)

    no_others = live_reduced[live_reduced["name"].apply(
        lambda n: song_to_genres.get(str(n).split(" - ")[-1].replace(".wav.csv", "").strip(), ["Other"]) != ["Other"]
    )]

    suggester_gen = model.suggest(no_others)
    return model, suggester_gen


@app.cell(hide_code=True)
def _(get_suggested, mo):
    mo.md(get_suggested())
    return


@app.cell(hide_code=True)
def _(mo):
    get_suggested, set_suggested = mo.state("No songs suggested yet. Press the 'next song' button.")
    score = mo.ui.slider(start=0, stop=1, step=0.01, value=0.5, label="Your score")

    score
    return get_suggested, score, set_suggested


@app.cell(hide_code=True)
def _(display_suggestion, mo):
    next = mo.ui.button(on_click=display_suggestion, label="Next song")
    next
    return


@app.cell(hide_code=True)
def _(score, set_suggested, song_to_genres, suggester_gen):
    def display_suggestion(_):
        suggestion = suggester_gen.__next__()
        genres = song_to_genres.get(suggestion.split(' - ')[-1].strip(), ["Other"])

        suggestion += f" ({', '.join(genres)})"
        set_suggested(f"Suggested song: {suggestion}")

        suggester_gen.send(score.value)
    return (display_suggestion,)


@app.cell(hide_code=True)
def _(get_suggested, mo, model, pd, reduced, song_to_genres):
    import plotly.express as px

    from plots import get_t, set_t, get_value_fig, set_value_fig
    _ = get_suggested()

    def update_fig():  # Only update plot one new step (after n suggestions and model update)
        if model.t == get_t():
            return get_value_fig()

        song_values = model._calc_song_values(reduced, over_all=True)
        expanded_rows = []
        for song, value in song_values.items():
            for genre in song_to_genres.get(song.split(" - ")[-1].replace(".wav.csv", "").strip(), []):
                expanded_rows.append({"genre": genre, "song": song.replace(".wav.csv", ""), "value": value})
        set_value_fig(px.box(pd.DataFrame(expanded_rows)[["value", "genre"]], 
                           x="genre", y="value", 
                           color="genre", title="Predicted song values by genre"))

        set_t(model.t)
        return get_value_fig()


    mo.ui.plotly(update_fig())
    return


if __name__ == "__main__":
    app.run()
