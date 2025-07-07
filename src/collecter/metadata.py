from logger import LOGGER

import traceback
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

from models import QueueJukeMIR, QueueAuditus
from db import get_session


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
    artist_name = sp.artist(artist_id)["name"]

    LOGGER.debug(f"Using top song '{top_song}' by '{artist_name}' for Last.fm lookup")

    lastfm_artist = lastfm.get_track(artist=artist_name, title=top_song).get_artist()
    LOGGER.info(f"Converted '{artist_name}' to Last.fm: {lastfm_artist.get_name()}")
    
    return lastfm_artist

def _sp_to_mb(artist_id: str) -> str:
    top_song = sp.artist_top_tracks(artist_id)["tracks"][0]["name"]
    artist_name = sp.artist(artist_id)["name"]
    LOGGER.debug(f"Searching MusicBrainz for: {artist_name} - {top_song}")

    res = mb.search_recordings(f"{sp.artist(artist_id)["name"]} - {top_song}")
    mb_name = res["recording-list"][0]["artist-credit"][0]["artist"]["name"]
    mb_id = res["recording-list"][0]["artist-credit"][0]["artist"]["id"]

    LOGGER.info(f"Found MusicBrainz match for '{artist_name}': {mb_name} ({mb_id})")

    return mb_id


def _lastfm_to_sp(artist: Artist) -> str:
    top_song = artist.get_top_tracks()[0].item.get_title()
    artist_name = artist.get_name()
    query = f"{artist_name} - {top_song}"
    res = sp.search(query, type="track")

    LOGGER.debug(f"Searching Spotify for: {query}")

    res = sp.search(query, type="track")
    spotify_artist = res["tracks"]["items"][0]["artists"][0]
    
    LOGGER.debug(f"Found Spotify match for {artist_name}: {spotify_artist["name"]}")
    
    return spotify_artist["id"]
   
def get_similar_artists(spotify_artist_id: str, degrees=1) -> list[str]:
    LOGGER.info(f"Getting similar artists for {spotify_artist_id}, {degrees} degrees.")
    assert degrees > 0

    similar = [_lastfm_to_sp(a.item) 
               for a in _sp_to_lastfm(spotify_artist_id).get_similar(limit=3)]

    LOGGER.debug(f"Found {len(similar)} similar artists: {similar}")

    if degrees == 1:
        return similar

    return [get_similar_artists(artist, degree-1) for artist in similar]


def _sp_collect_results(chunk: dict) -> list[str]:
    items = chunk["items"]

    while chunk.get("next"):
        chunk = sp.next(chunk)
        items.extend(chunk["items"])
    
    return items

def _get_sp_playlist_tracks(playlist_id: str) -> list[str]:
    items = _sp_collect_results(sp.playlist(playlist_id)["tracks"])
    track_ids = [track["track"]["id"] for track in items]
    
    LOGGER.debug(f"Found {len(track_ids)} tracks in playlist {playlist_id}")
    return track_ids

def _get_sp_user_playlist() -> list[str]:
    items = _sp_collect_results(sp.current_user_playlists(limit=50))
    playlist_ids = [playlist["id"] for playlist in items]
    
    LOGGER.info(f"Found {len(playlist_ids)} user playlists")
    return playlist_ids

def _get_sp_liked_tracks() -> list[str]:
    items = _sp_collect_results(sp.current_user_saved_tracks(limit=50))
    track_ids = [track["track"]["id"] for track in items]
    
    LOGGER.info(f"Found {len(track_ids)} liked tracks")
    return track_ids

def _get_sp_album_tracks(album_id: str) -> list[str]:
    items = _sp_collect_results(sp.album_tracks(album_id, limit=50))
    album_ids = [album["id"] for album in items]
    
    LOGGER.debug(f"Found {len(album_ids)} tracks in album {album_id}")
    return album_ids

def _get_sp_artist_albums(artist_id: str) -> list[str]:
    items = _sp_collect_results(sp.artist_albums(artist_id, limit=50))
    album_ids = [album["id"] for album in items]
    
    LOGGER.debug(f"Found {len(album_ids)} albums for artist {artist_id}")
    return album_ids

def get_sp_artist_tracks(artist_id: str) -> list[str]:
    albums = _get_sp_artist_albums(artist_id)
    tracks = [track for album in albums for track in _get_sp_album_tracks(album)]
    
    LOGGER.info(f"Found {len(tracks)} total tracks for artist {artist_id}")
    return tracks


def _add_to_db_queue(spotify_track_ids: list[str]):
    with get_session() as s:
        # TODO: If associated song is already processed (in Songs table): skip.

        for track_id in set(spotify_track_ids):  # Set to skip duplicates.
            s.add(QueueJukeMIR(spotify_id=track_id))
            s.add(QueueAuditus(spotify_id=track_id))
        s.commit()

# TODO: Some kind of live/remastered filter? Those are effectively duplicates (in most cases).

def queue_sp_user():  # TODO: Take userID instead of current user.
    LOGGER.info("Queueing user's Spotify library")
    liked = _get_sp_liked_tracks()
    playlist_ids = _get_sp_user_playlist()
    
    LOGGER.debug(f"Processing {len(playlist_ids)} playlists")
    playlist_tracks = [track for p in playlist_ids
                       for track in _get_sp_playlist_tracks(p)]
    
    _add_to_db_queue(liked + playlist_tracks)

def queue_sp_history(history: list[dict]):
    LOGGER.info(f"Queueing {len(history)} tracks from listening history")
    song_ids = [listen["spotify_track_uri"].split(":")[-1] for listen in history]
    _add_to_db_queue(song_ids)

def simple_queue_new_music():
    LOGGER.info("Queueing new music based on similar artists")
    # get known artists from DB
    similar = [get_similar_artists(a) for a in artists]
    LOGGER.debug(f"Found similar artists for {len(similar)} known artists")
    
    tracks = [t for t in get_sp_artist_tracks(a) for a in similar]
    LOGGER.info(f"Queueing {len(tracks)} tracks from similar artists")
    _add_to_db_queue(tracks)

def _get_mb_artist_tags(musicbrainz_id: str | None) -> list[str]:
    # We could also use MBID for this but on the very first artist (CAN) I tried it was already wrong.
    if not musicbrainz_id: return []

    try:
        res = mb.get_artist_by_id(artist_id, includes=["tags", "user-tags"])
        return [x["name"].lower() for x in res["artist"].get("tag-list", [])]
    except:
        # LOGGER.warning(f"Something went wrong with the request: {traceback.format_exc()}")
        return []
    
def push_artist_metadata(spotify_id: str) -> dict:
    lfm_tags = _sp_to_lastfm(spotify_id).get_top_tags()
    return {"LastFM": [tag.item.get_name().lower() for tag in lfm_tags],
            "MusicBrainz": _get_mb_artist_tags(_sp_to_mb(spotify_id))}

def push_track_metadata(spotify_id: str) -> dict:
    track = sp.track(spotify_id)
    mb_track = mb.search_recordings(f"{track["artists"][0]["name"]} - {track["name"]}")
    mb_tags = mb.get_recording_by_id(mb_track["recording-list"][0]["id"],
                                     includes=["tags", "user-tags"])

    lfm_tags = lastfm.get_track(artist=track["artists"][0]["name"],
                                title=track["name"]).get_top_tags()

    return {"LastFM": [tag.item.get_name().lower() for tag in lfm_tags],
            "MusicBrainz": [x["name"].lower() for x in mb_tags["recording"].get("tag-list", [])]}
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


