from logger import LOGGER
import os

import pylast
from pylast import Tag, Artist, Track
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# import musicbrainzngs


SCOPES = ['user-library-read',
          'playlist-read-private',
          'playlist-read-collaborative']

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SCOPES,
                                               client_id=os.environ.get("SPOTIFY_CLIENT_ID"),
                                               client_secret=os.environ.get("SPOTIFY_CLIENT_SECRET"),
                                               redirect_uri='http://localhost:8888/callback'))

lastfm = pylast.LastFMNetwork(
    api_key=os.environ.get("LASTFM_API_KEY"),
    api_secret=os.environ.get("LASTFM_API_SECRET"),
    username=os.environ.get("LASTFM_USERNAME"),
    password_hash=os.environ.get("LASTFM_PW")
)


def _spotify_to_lastfm(artist_id: str) -> Artist:
    """
    Simple name querying can be unreliable. Maybe replace with smth via top song matching?
        Song - artist combo is a much more unique identifier than just song name.
            Can even try n top songs in case some aren't in LastFM DB.
            Universal IDs like ISRC or EAN don't have enough coverage.
                - maybe ID with fallback system?
    """
    name = sp.artist(artist_id)["name"]
    
    return lastfm.get_artist(name)
    

def _lastfm_to_spotify(artist: Artist) -> str:
    sp_artist = sp.search(artist.get_name(), type="artist")["artists"]["items"][0]
    return sp_artist["id"]
   
def get_similar_artists(spotify_artist_id: str, degrees=1) -> List[str]:
    assert degrees > 0
    similar = [_lastfm_to_spotify(a) 
               for a in _spotify_to_lastfm(spotify_artist_id).get_similar()]

    if degrees == 1:
        return similar

    return [get_similar_artists(artist, degree-1) for artist in similar]


def _spotify_collect_results(spotify_chunk: dict) -> list[str]:
    items = chunk["items"]

    while chunk.get("next"):
        chunk = sp.next(chunk)
        items.extend(chunk["items"])
    
    return [x["id"] for x in items]

def get_spotify_playlist_tracks(playlist_id: str) -> list[str]:
    return _spotify_collect_results(sp.playlist(playlist_id, limit=50)["tracks"])

def _get_spotify_album_tracks(album_id: str) -> list[str]:
    return _spotify_collect_results(sp.albums_tracks(album_id, limit=50)["tracks"])

def _get_spotify_artist_albums(artist_id: str) -> list[str]:
    return _spotify_collect_results(sp.artist_albums(artist_id, limit=50)["tracks"])

def get_spotify_artist_tracks(artist_id: str):
    return [_get_spotify_album_tracks(album) for album in _get_spotify_artist_albums(artist_id)]


def queue_spotify_user():
    pass
    # Liked tracks
    # playlists

def queue_spotify_history(history: list[dict]):
    songs = [listen["spotify_track_uri"].split(":")[-1] for listen in history]
    # Push to db queue

def simple_queue_music():
    # get known artists from DB
    similar = [get_similar_artists(a) for a in artists]
    
    tracks = [t for get_spotify_artist_tracks(a) for a in similar]
    # Push to DB queue

    
# LastFM
# Track -> tags
# Track -> similar tracks
# Artist -> tags
# Artist -> similar artists
# Tag -> description (cool for some future feature)
# Tag -> similar tags

# OUTPUT
# Spotify Artist -> Related spotify artists

# WE CAN BUILD (relational):
    # artist graph
    # tag graph
        # And somehow relate to our data (so we know which tags/artists/etc to downlaod more music from)

