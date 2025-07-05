from logger import LOGGER
import os

from dotenv import load_dotenv
load_dotenv()

import pylast
from pylast import Tag, Artist, Track
import spotipy
from spotipy.oauth2 import SpotifyOAuth

import musicbrainzngs as mb
mb.set_useragent("Suggest: Music Recommender", "1.0", contact=os.environ["EMAIL_ADDR"])
mb.set_rate_limit()
mb.auth(os.environ["MB_USERNAME"], os.environ["MB_PW"])

SCOPES = ['user-library-read',
          'playlist-read-private',
          'playlist-read-collaborative']

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SCOPES,
                                               client_id=os.environ["SPOTIFY_CLIENT_ID"],
                                               client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
                                               redirect_uri='http://localhost:8888/callback'))

lastfm = pylast.LastFMNetwork(
    api_key=os.environ["LASTFM_API_KEY"],
    api_secret=os.environ["LASTFM_API_SECRET"],
    username=os.environ["LASTFM_USERNAME"],
    password_hash=pylast.md5(os.environ["LASTFM_PW"])
)


def _sp_to_lastfm(artist_id: str) -> Artist:
    """
    Song - artist combo is a much more unique identifier than just song name.
        Universal IDs like ISRC or EAN don't have enough coverage.
            - maybe ID with fallback system?

        If the top song happens to be featuring another artist, 
            it's a 50/50 whether we get the right one. 
        Compare artists on top N songs for more certainty?
            - or will a simple name artist name check work for such cases?
    """

    top_song = sp.artist_top_tracks(artist_id)["tracks"][0]["name"]
    return lastfm.get_track(artist=sp.artist(artist_id)["name"],
                             title=top_song).get_artist()

def _sp_to_mb(artist_id: str) -> str:
    top_song = sp.artist_top_tracks(artist_id)["tracks"][0]["name"]

    res = mb.search_recordings(f"{sp.artist(artist_id)["name"]} - {top_song}")

    return res["recording-list"][0]["artist-credit"][0]["artist"]["id"]


def _lastfm_to_sp(artist: Artist) -> str:
    top_song = a.get_top_tracks()[0].item.get_title()
    res = sp.search(f"{artist.get_name()} - {top_song}", type="track")

    return res["tracks"]["items"][0]["artists"][0]["name"]
   
def get_similar_artists(spotify_artist_id: str, degrees=1) -> List[str]:
    assert degrees > 0
    similar = [_lastfm_to_sp(a) 
               for a in _sp_to_lastfm(spotify_artist_id).get_similar()]

    if degrees == 1:
        return similar

    return [get_similar_artists(artist, degree-1) for artist in similar]


def _sp_collect_results(spotify_chunk: dict) -> list[str]:
    items = chunk["items"]

    while chunk.get("next"):
        chunk = sp.next(chunk)
        items.extend(chunk["items"])
    
    return [x["id"] for x in items]

def _get_sp_playlist_tracks(playlist_id: str) -> list[str]:
    return _sp_collect_results(sp.playlist(playlist_id, limit=50)["tracks"])

def _get_sp_user_playlist() -> list[str]:
    return _sp_collect_results(sp.current_user_playlists(limit=50)["tracks"])

def _get_sp_liked_tracks() -> list[str]:
    return _sp_collect_results(sp.current_user_saved_tracks(limit=50)["tracks"])

def _get_sp_album_tracks(album_id: str) -> list[str]:
    return _sp_collect_results(sp.albums_tracks(album_id, limit=50)["tracks"])

def _get_sp_artist_albums(artist_id: str) -> list[str]:
    return _sp_collect_results(sp.artist_albums(artist_id, limit=50)["tracks"])

def get_sp_artist_tracks(artist_id: str) -> list[str]:
    return [_get_sp_album_tracks(album) for album in _get_spotify_artist_albums(artist_id)]

def _add_to_db_queue(spotify_track_ids: list[str]):
    for track_id in set(spotify_track_ids):  # Set to skip duplicates.
        pass
        # If associated song is already processed (in Songs table): skip.

        # push to db


def queue_sp_user():
    liked = _get_sp_liked_tracks()
    playlists_tracks = [_get_sp_playlist_tracks(p) for p in _get_spotify_user_playlist()]
    
    _add_to_db_queue(liked + playlist_tracks)

def queue_sp_history(history: list[dict]):
    song_ids = [listen["spotify_track_uri"].split(":")[-1] for listen in history]
    _add_to_db_queue(song_ids)

def simple_queue_new_music():
    # get known artists from DB
    similar = [get_similar_artists(a) for a in artists]
    
    # tracks = [t for get_sp_artist_tracks(a) for a in similar]
    _add_to_db_queue([t for get_sp_artist_tracks(a) for a in similar])

def _get_mb_artist_tags(musicbrainz_id: str | None) -> list[str]:
    # We could also use MBID for this but on the very first artist (CAN) I tried it was already wrong.
    if not musicbrainz_id: return []

    try:
        res = mb.get_artist_by_id(artist_id, includes=["tags", "user-tags"])
        return [x["name"] for x in res["artist"]["tag-list"]]
    except WebServiceError as exc:
        LOGGER.WARNING("Something went wrong with the request: %s" % exc)
        return []
    
def get_artist_metadata(spotify_id: str) -> dict:
    return {"LastFM": _sp_to_lastfm(spotify_id).get_top_tags(),
            "MusicBrainz": _get_mb_artist_tags(_sp_to_mb(spotify_id))}

def get_track_metadata(spotify_id: str) -> dict:
    res = sp.search(f"{artist.get_name()} - {top_song}", type="track")

    track = sp.track(spotify_id)
    mb_track = mb.search_recordings(f"{track["artists"][0]["name"]} - {track["name"]}")
    mb_tags = mb.get_recording_by_id(x["recording-list"][0]["id"],
                                     includes=["tags", "user-tags"])
    return {"LastFM": lastfm.get_track(artist=track["artists"][0]["name"],
                                       title=track["name"]).get_top_tags(),
            "MusicBrainz": [x["name"] for x in mb_tags["artist"]["tag-list"]]}
# LastFM
# - Track -> tags
# Track -> similar tracks
# - Artist -> tags
# - Artist -> similar artists
# Tag -> description (cool for some future feature)
# Tag -> similar tags

# WE CAN BUILD (relational):
    # artist graph
    # tag graph
        # And somehow relate to our data (so we know which tags/artists/etc to downlaod more music from)

