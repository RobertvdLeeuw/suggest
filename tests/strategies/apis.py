from hypothesis import strategies as st
from collecter.embedders import QueueObject
from models import Song, Artist, User, Listen, ListenChunk, SongMetadata, MetadataType, StartEndReason


@st.composite
def spotify_id_strat(draw):
    return data.draw(st.from_regex("[a-zA-Z0-9]{22}"))


@st.composite
def song_strat(draw, artists=None):
    """Generate Song objects with optional artist relationships."""

    spotify_id = draw(spotify_id_strat())
    song_name = draw(st.text(min_size=1, max_size=100))
    
    return Song(
        spotify_id=spotify_id,
        song_name=song_name,
        artists=artists or []
    )

@st.composite 
def artist_strat(draw):
    spotify_id = draw(spotify_id_strat())
    artist_name = draw(st.text(min_size=1, max_size=100))
    
    return Artist(
        spotify_id=spotify_id,
        artist_name=artist_name,
        entirely_queued=False,
        similar_queued=False
    )

@st.composite
def user_strat(draw):
    spotify_id = draw(spotify_id_strat())
    username = draw(st.text(min_size=1, max_size=100))
    
    return User(
        spotify_id=spotify_id,
        username=username
    )

@st.composite
def listen_strat(draw, user_id=None, song_id=None):
    return Listen(
        user_id=user_id or draw(spotify_id_strat()),
        song_id=song_id or draw(spotify_id_strat()),
        listened_at=draw(st.datetimes()),
        ms_played=draw(st.integers(min_value=1000, max_value=300000)),
        reason_start=draw(st.sampled_from([r.value for r in StartEndReason])),
        reason_end=draw(st.sampled_from([r.value for r in StartEndReason])),
        from_history=draw(st.booleans())
    )

