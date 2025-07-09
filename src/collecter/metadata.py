from logger import LOGGER

import traceback
import os

from dotenv import load_dotenv
load_dotenv()

import pylast
from pylast import Tag, Artist, Track
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from models import EmbeddingJukeMIR, EmbeddingAuditus

SCOPES = ['user-library-read',
          'playlist-read-private',
          'playlist-read-collaborative']

sp_oauth = SpotifyOAuth(
    client_id=os.environ["SPOTIFY_CLIENT_ID"],
    client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
    redirect_uri="http://127.0.0.1:8080/callback",
    scope=SCOPES
)

token_info = sp_oauth.refresh_access_token(os.environ["SPOTIFY_REFRESH_TOKEN"])
sp = spotipy.Spotify(auth=token_info['access_token'])

import musicbrainzngs as mb
mb.set_useragent("Suggest: Music Recommender", "1.0", contact=os.environ["EMAIL_ADDR"])
mb.set_rate_limit()
mb.auth(os.environ["MB_USERNAME"], os.environ["MB_PW"])

from models import QueueJukeMIR, QueueAuditus, Song, SongMetadata, Artist, ArtistMetadata, MetadataType
from db import get_session
from sqlalchemy import select, exists
from sqlalchemy.sql.expression import func


lastfm = pylast.LastFMNetwork(
    api_key=os.environ["LASTFM_API_KEY"],
    api_secret=os.environ["LASTFM_API_SECRET"],
    username=os.environ["LASTFM_USERNAME"],
    password_hash=pylast.md5(os.environ["LASTFM_PW"])
)


def _sp_to_lastfm(artist_id: str) -> Artist | None:
    """
    Song - artist combo is a much more unique identifier than just song name.
        Universal IDs like ISRC or EAN don't have enough coverage.
            - maybe ID with fallback system?
    """

    top_song = sp.artist_top_tracks(artist_id)["tracks"][0]["name"]
    artist_name = sp.artist(artist_id)["name"]

    LOGGER.debug(f"Using top song '{top_song}' by '{artist_name}' for Last.fm lookup")

    lastfm_artist = lastfm.get_track(artist=artist_name, title=top_song).get_artist()
    LOGGER.info(f"Converted '{artist_name}' to Last.fm: {lastfm_artist.get_name()}")
    
    return lastfm_artist

def _sp_to_mb(artist_id: str) -> str | None:
    top_song = sp.artist_top_tracks(artist_id)["tracks"][0]["name"]
    artist_name = sp.artist(artist_id)["name"]
    LOGGER.debug(f"Searching MusicBrainz for: {artist_name} - {top_song}")

    res = mb.search_recordings(f"{sp.artist(artist_id)["name"]} - {top_song}")
    for mb_artist in res["recording-list"][0]["artist-credit"]:
        if artist_name.strip().lower() == mb_artist["artist"]["name"].strip().lower():
            mb_name = mb_artist["artist"]["name"]
            mb_id = mb_artist["artist"]["id"]
            break
    else:
        artist_names = [a["artist"]["name"] for a in res["recording-list"][0]["artist-credit"]]
        LOGGER.warning(f"No MusicBrainz match found for {artist_name} among: {", ".join(artist_names)}")
        return

    LOGGER.info(f"Found MusicBrainz match for '{artist_name}': {mb_name} ({mb_id})")
    return mb_id


def _lastfm_to_sp(artist: Artist) -> str | None:
    top_song = artist.get_top_tracks()[0].item.get_title()
    artist_name = artist.get_name()

    LOGGER.debug(f"Searching Spotify for: {artist_name} - {top_song}")

    res = sp.search(f"{artist_name} - {top_song}", type="track")
    for sp_artist in res["tracks"]["items"][0]["artists"]:
        if artist_name.strip().lower() == sp_artist.strip().lower():
            spotify_artist = sp_artist
            break
    else:
        artist_names = [a["name"] for a in res["tracks"]["items"][0]["artists"]]
        LOGGER.warning(f"No Spotify match found for {artist_name} among: {", ".join(artist_names)}")
        return
    
    LOGGER.debug(f"Found Spotify match for {artist_name}: {spotify_artist["name"]}")
    
    return spotify_artist["id"]
   
def get_similar_artists(spotify_artist_id: str, degrees=1) -> list[str]:
    LOGGER.info(f"Getting similar artists for {spotify_artist_id}, {degrees} degrees.")
    assert degrees > 0

    similar = [_lastfm_to_sp(a.item) 
               for a in _sp_to_lastfm(spotify_artist_id).get_similar(limit=3)]
    similar = [s for s in similar if x is not None]

    LOGGER.debug(f"Found {len(similar)} similar artists: {similar}")

    if degrees == 1:
        return similar

    return [get_similar_artists(artist, degree-1) for artist in similar]


def _sp_collect_results(chunk: dict, data_type: str) -> list[str]:
    items = chunk["items"]

    i = 0
    while chunk.get("next"):
        i += 1

        chunk = sp.next(chunk)
        items.extend(chunk["items"])
        LOGGER.debug(f"Collecting {data_type}, chunk {i}.")
    
    return items

def _get_sp_playlist_tracks(playlist_id: str) -> list[str]:
    LOGGER.debug(f"Getting tracks of playlist {"spotify_id"}.")

    items = _sp_collect_results(sp.playlist(playlist_id)["tracks"], "playlist tracks")
    track_ids = [track["track"]["id"] for track in items]
    
    LOGGER.info(f"Found {len(track_ids)} tracks in playlist {playlist_id}")
    return track_ids

def _get_sp_user_playlist() -> list[str]:
    LOGGER.debug(f"Getting user playlists.")

    items = _sp_collect_results(sp.current_user_playlists(limit=50), "user playlists")
    playlist_ids = [playlist["id"] for playlist in items]
    
    LOGGER.info(f"Found {len(playlist_ids)} user playlists")
    return playlist_ids

def _get_sp_liked_tracks() -> list[str]:
    LOGGER.debug(f"Getting liked user tracks.")

    items = _sp_collect_results(sp.current_user_saved_tracks(limit=50), "user liked")
    track_ids = [track["track"]["id"] for track in items]
    
    LOGGER.info(f"Found {len(track_ids)} liked tracks")
    return track_ids

def _get_sp_album_tracks(album_id: str) -> list[str]:
    LOGGER.debug(f"Getting tracks of album {"spotify_id"}.")

    items = _sp_collect_results(sp.album_tracks(album_id, limit=50), "album tracks")
    album_ids = [album["id"] for album in items]
    
    LOGGER.info(f"Found {len(album_ids)} tracks in album {album_id}")
    return album_ids

def _get_sp_artist_albums(artist_id: str) -> list[str]:
    LOGGER.debug(f"Getting albums of artist {artist_id}.")

    items = _sp_collect_results(sp.artist_albums(artist_id, limit=50), "artist albums")
    album_ids = [album["id"] for album in items]
    
    LOGGER.info(f"Found {len(album_ids)} albums for artist {artist_id}")
    return album_ids

def get_sp_artist_tracks(artist_id: str) -> list[str]:
    LOGGER.debug(f"Getting tracks of artist {artist_id}.")

    # TODO: singles too (but prevent overlap/duplicates)
    albums = _get_sp_artist_albums(artist_id)
    tracks = [track for album in albums for track in _get_sp_album_tracks(album)]
    
    LOGGER.info(f"Found {len(tracks)} total tracks for artist {artist_id}")
    return tracks


async def check_in_table(table, col, value, session=None):
    if session is None:
        session = await get_session()

    async with session:
        result = await session.execute(select(table).where(col == value))
        # result = await session.execute(select(exists().where(col == value)))
        return result.scalar_one_or_none()  # TODO: Does this just grab first for multiple results?

QUEUE_OBJECTS = [(EmbeddingJukeMIR, QueueJukeMIR, "JukeMIR"), 
                 (EmbeddingAuditus, QueueAuditus, "Auditus")]
async def _add_to_db_queue(spotify_track_ids: list[str]):
    LOGGER.debug(f"Adding {len(spotify_track_ids)} songs to queue.")

    async with get_session() as s:
        for track_id in set(spotify_track_ids): 
            for emb_type, q_type, name in QUEUE_OBJECTS:
                if await check_in_table(q_type, q_type.spotify_id, track_id, s):
                    LOGGER.debug(f"Song {track_id} already queued for {name}.")
                    continue


                if song := await check_in_table(Song, Song.spotify_id, track_id, s):
                    if await check_in_table(emb_type, emb_type.song_id, song.song_id, s):
                        LOGGER.debug(f"Song {track_id} already embedded for {name}.")
                        continue

                LOGGER.debug(f"Pushing song {track_id} to {name} queue.")
                s.add(q_type(spotify_id=track_id))
        await s.commit()
    LOGGER.info(f"Added {len(spotify_track_ids)} songs to queue.")

# TODO: Some kind of live/remastered filter? Those are effectively duplicates (in most cases).

async def queue_sp_user():  # TODO: Take userID instead of current user.
    LOGGER.info("Queueing user's Spotify library")
    liked = _get_sp_liked_tracks()
    playlist_ids = _get_sp_user_playlist()
    
    LOGGER.debug(f"Processing {len(playlist_ids)} playlists")
    playlist_tracks = [track for p in playlist_ids
                       for track in _get_sp_playlist_tracks(p)]
    
    await _add_to_db_queue(liked + playlist_tracks)

async def queue_sp_history(history: list[dict]):
    LOGGER.info(f"Queueing {len(history)} tracks from listening history")
    song_ids = [listen["spotify_track_uri"].split(":")[-1] for listen in history]
    await _add_to_db_queue(song_ids)

async def simple_queue_new_music():
    LOGGER.info("Queueing new music based on similar artists")
    
    async with get_session() as s:
        artists = await s.execute(select(Artist.spotify_id)
                                  .order_by(func.random())
                                  .limit(10))

        artists = artists.scalars().all()

    similar = [get_similar_artists(a) for a in artists]
    LOGGER.debug(f"Found similar artists for {len(similar)} known artists")
    
    tracks = [t for a in similar for t in get_sp_artist_tracks(a)]
    LOGGER.info(f"Queueing {len(tracks)} tracks from similar artists")
    await _add_to_db_queue(tracks)

    for a in artists:
        a.similar_queued = True

    async with get_session() as s:
        s.add_all(artists)
        await s.commit()


def _get_mb_artist_tags(musicbrainz_id: str | None) -> list[str]:
    # We could also use MBID for this but on the very first artist (CAN) I tried it was already wrong.
    if not musicbrainz_id: return []

    try:
        res = mb.get_artist_by_id(artist_id, includes=["tags", "user-tags"])
        return [x["name"].lower() for x in res["artist"].get("tag-list", [])]
    except:
        # LOGGER.warning(f"Something went wrong with the request: {traceback.format_exc()}")
        return []
    
async def create_push_artist(spotify_id: str) -> Artist:
    artist_name = sp.artist(spotify_id)["name"]

    async with get_session() as s:
        artist.extra_data = []
        if lfm_artist := _sp_to_lastfm(spotify_id):
            LOGGER.debug(f"Adding LastFM tags to artist '{artist_name}'.")
            artist.extra_data.extend(
                [ArtistMetadata(type=MetadataType.genre, 
                                value=tag.item.get_name().strip().capitalize(), 
                                source="LastFM")
                    for tag in lfm_artist.get_top_tags()]
            )

        if mb_artist := _get_mb_artist_tags(_sp_to_mb(spotify_id)):
            LOGGER.debug(f"Adding MusicBrainz tags to artist '{artist_name}'.")
            artist.extra_data.extend(
                [ArtistMetadata(type=MetadataType.genre, 
                                value=tag.item.get_name().lower(), 
                                source="MusicBrainz")
                    for tag in mb_artist]
            )

        s.add(artist)
        await s.commit()

    LOGGER.info(f"Successfully created artist '{artist_name}'.")
    return artist

# TODO: Decorator for try-except for common DB errors.
async def push_track_metadata(spotify_id: str) -> Song:
    track = sp.track(spotify_id)

    async with get_session() as s:
        if song := await check_in_table(Song, Song.spotify_id, spotify_id, s):
            LOGGER.info(f"Song {track["name"]} already exists")
            return song

        artists = []
        for artist_data in track["artists"]:
            artist = await check_in_table(Artist, Artist.spotify_id, artist_data["id"], s)

            if artist is None:
                LOGGER.debug(f"Creating new artist: {artist["name"]}'.")
                artist = await create_push_artist(artist["id"])
            
            artists.append(artist)

        song = Song(spotify_id=spotify_id, 
                    song_name=track["name"],
                    artists=artists)

        song.extra_data = []
        if lfm_song := lastfm.get_track(artist=track["artists"][0]["name"], title=track["name"]):
            LOGGER.debug(f"Adding LastFM tags to track '{track["name"]}'.")
            song.extra_data.extend(
                [SongMetadata(type=MetadataType.genre, 
                              value=tag.item.get_name().strip().capitalize(), 
                              source="LastFM")
                    for tag in lfm_song.get_top_tags()] 
            )

        if mb_song := mb.search_recordings(f"{track["artists"][0]["name"]} - {track["name"]}"):
            LOGGER.debug(f"Adding MusicBrainz tags to track '{track["name"]}'.")
            song.extra_data.extend(
                [SongMetadata(type=MetadataType.genre, 
                              value=tag.item.get_name().lower(), 
                              source="MusicBrainz")
                    for tag in mb.get_recording_by_id(mb_song["recording-list"][0]["id"],
                                                      includes=["tags", "user-tags"])]
            )

        s.add(song)
        await s.commit()

    LOGGER.info(f"Successfully created song '{song.song_name}'.")
    return song


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


