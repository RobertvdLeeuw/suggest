from logger import LOGGER

import traceback
import asyncio
import os
import time

from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import pylast
from pylast import Tag, Artist, Track
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from models import EmbeddingJukeMIR, EmbeddingAuditus, Artist

from requests.exceptions import ReadTimeout

SCOPES = ['user-library-read',
          'playlist-read-private',
          'playlist-read-collaborative',
          'user-read-currently-playing',
          'user-read-playback-state']

sp_oauth = SpotifyOAuth(
    client_id=os.environ["SPOTIFY_CLIENT_ID"],
    client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
    redirect_uri="http://127.0.0.1:8888/callback",
    scope=SCOPES
)


token_info = sp_oauth.refresh_access_token(os.environ["SPOTIFY_REFRESH_TOKEN"])
sp = spotipy.Spotify(auth=token_info['access_token'])

def get_spotipy():
    return sp

def refresh_spotipy():
    global sp

    LOGGER.info("Refreshing Spotipy client access token.")
    token_info = sp_oauth.refresh_access_token(os.environ["SPOTIFY_REFRESH_TOKEN"])
    sp = spotipy.Spotify(auth=token_info['access_token'])
    LOGGER.info("Refreshed Spotipy client access token.")

import musicbrainzngs as mb
mb.set_useragent("Suggest: Music Recommender", "1.0", contact=os.environ["EMAIL_ADDR"])
mb.set_rate_limit()
mb.auth(os.environ["MB_USERNAME"], os.environ["MB_PW"])

from models import (
    QueueJukeMIR, QueueAuditus,
    Song, SongMetadata, 
    Artist, ArtistMetadata, MetadataType, 
    ListenChunk, Listen,
    User
)
from db import get_session
from sqlalchemy import select, exists
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.sql.expression import func


lastfm = pylast.LastFMNetwork(
    api_key=os.environ["LASTFM_API_KEY"],
    api_secret=os.environ["LASTFM_API_SECRET"],
    username=os.environ["LASTFM_USERNAME"],
    password_hash=pylast.md5(os.environ["LASTFM_PW"])
)
lastfm.enable_rate_limit()


def _sp_to_lastfm(artist_id: str) -> Artist | None:
    """
    Song - artist combo is a much more unique identifier than just song name.
        Universal IDs like ISRC or EAN don't have enough coverage.
            - maybe ID with fallback system?
    """
    if artist_id is None: return None
    # time.sleep(0.25)  # Prevent rate-limit.

    try:
        top_song = sp.artist_top_tracks(artist_id)["tracks"][0]["name"]
    except:
        LOGGER.error(F"No top track for artist '{artist_id}': {sp.artist_top_tracks(artist_id)}")
        return

    artist_name = sp.artist(artist_id)["name"]

    LOGGER.debug(f"Using top song '{top_song}' by '{artist_name}' for Last.fm lookup")

    lastfm_artist = lastfm.get_track(artist=artist_name, title=top_song).get_artist()
    LOGGER.info(f"Converted '{artist_name}' to Last.fm: {lastfm_artist.get_name()}")
    
    return lastfm_artist

def _sp_to_mb(artist_id: str) -> str | None:
    if artist_id is None: return None
    # time.sleep(0.25)  # Prevent rate-limit.

    try:
        top_song = sp.artist_top_tracks(artist_id)["tracks"][0]["name"]
    except:
        LOGGER.error(F"No top track for artist '{artist_id}': {sp.artist_top_tracks(artist_id)}")
        return
    artist_name = sp.artist(artist_id)["name"]
    LOGGER.info(f"Searching MusicBrainz for: {artist_name} - {top_song}")

    res = mb.search_recordings(f'artist:"{artist_name}" AND recording:"{top_song}"')

    if len(res["recording-list"]) == 0:
        LOGGER.warning(f"No artists found in response for '{artist_name} - {top_song}': {res}")
        return None

    for mb_artist in res["recording-list"][0]["artist-credit"]:
        if not isinstance(mb_artist, dict) or "artist" not in mb_artist:
            LOGGER.debug(f"Skipping MB artist entry '{mb_artist}'.")
            continue

        if artist_name.strip().lower() == mb_artist["artist"]["name"].strip().lower():
            mb_name = mb_artist["artist"]["name"]
            mb_id = mb_artist["artist"]["id"]
            break
    else:
        artist_names = [a["artist"]["name"] for a in res["recording-list"][0]["artist-credit"]
                        if isinstance(a, dict) and "artist" not in a]
        LOGGER.warning(f"No MusicBrainz match found for {artist_name} " \
                       f"among: {", ".join(artist_names)}")
        return

    LOGGER.info(f"Found MusicBrainz match for '{artist_name}': {mb_name} ({mb_id})")
    return mb_id


def _lastfm_to_sp(artist: Artist) -> str | None:
    if artist is None: return None
    # time.sleep(0.25)  # Prevent rate-limit.

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

    lfm_artist = _sp_to_lastfm(spotify_artist_id)
    if lfm_artist is None: return []

    similar = [_lastfm_to_sp(a.item) 
               for a in lfm_artist.get_similar(limit=3)]
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
        time.sleep(0.5)  # Prevent rate-limit.
    
    return items

def _get_sp_playlist_tracks(playlist_id: str) -> list[str]:
    LOGGER.debug(f"Getting tracks of playlist {"spotify_id"}.")

    items = _sp_collect_results(sp.playlist(playlist_id)["tracks"], "playlist tracks")
    track_ids = [track["track"]["id"] for track in items if track["track"]["id"] is not None]
    
    LOGGER.info(f"Found {len(track_ids)} tracks in playlist {playlist_id}")
    return track_ids

def _get_sp_user_playlists() -> list[str]:
    LOGGER.debug(f"Getting user playlists.")

    items = _sp_collect_results(sp.current_user_playlists(limit=50), "user playlists")
    playlist_ids = [playlist["id"] for playlist in items if playlist["id"] is not None]
    
    LOGGER.info(f"Found {len(playlist_ids)} user playlists")
    return playlist_ids

def _get_sp_liked_tracks() -> list[str]:
    LOGGER.debug(f"Getting liked user tracks.")

    items = _sp_collect_results(sp.current_user_saved_tracks(limit=50), "user liked")
    track_ids = [track["track"]["id"] for track in items if track["track"]["id"] is not None]
    
    LOGGER.info(f"Found {len(track_ids)} liked tracks")
    return track_ids

def _get_sp_album_tracks(album_id: str) -> list[str]:
    LOGGER.debug(f"Getting tracks of album {"spotify_id"}.")

    items = _sp_collect_results(sp.album_tracks(album_id, limit=50), "album tracks")
    album_ids = [album["id"] for album in items if album["id"] is not None]
    
    LOGGER.info(f"Found {len(album_ids)} tracks in album {album_id}")
    return album_ids

def _get_sp_artist_albums(artist_id: str) -> list[str]:
    LOGGER.debug(f"Getting albums of artist {artist_id}.")

    items = _sp_collect_results(sp.artist_albums(artist_id, limit=50), "artist albums")
    album_ids = [album["id"] for album in items if album["id"] is not None]
    
    LOGGER.info(f"Found {len(album_ids)} albums for artist {artist_id}")
    return album_ids

def get_sp_artist_tracks(artist_id: str) -> list[str]:
    LOGGER.debug(f"Getting tracks of artist {artist_id}.")

    # TODO: singles too (but prevent overlap/duplicates)
    albums = _get_sp_artist_albums(artist_id)
    tracks = [track for album in albums for track in _get_sp_album_tracks(album)]
    
    LOGGER.info(f"Found {len(tracks)} total tracks for artist {artist_id}")
    return tracks


def pass_session_capable(func):
    async def inner(*args, **kwargs):
        if s := kwargs.get("session"):
            return await func(*args, **kwargs)

        async with get_session() as s:
            kwargs["session"] = s
            return await func(*args, **kwargs)

    return inner


@pass_session_capable
async def check_in_table(table, col, value, session=None):
    result = await session.execute(select(table).where(col == value).limit(1))
    item = result.scalar_one_or_none()
    LOGGER.debug(f"Checking {table.__name__} for {col.name}={value}: " \
                 f"{'Found' if item else 'Not found'}")
    return item


@pass_session_capable
async def check_multiple_in_table(table, col, values: list, session=None) -> dict:
    """
    Check multiple values at once for better performance.
    
    Returns:
        Dictionary mapping values to found objects (or None if not found)
    """
    if not values:
        return {}
    
    result = await session.execute(select(table).where(col.in_(values)))
    found_items = result.scalars().all()
    
    found_dict = {}
    for item in found_items:
        item_value = getattr(item, col.name)
        found_dict[item_value] = item
    
    # Fill in None for missing values
    return {value: found_dict.get(value) for value in values}

            
QUEUE_OBJECTS = [(EmbeddingJukeMIR, QueueJukeMIR, "JukeMIR"), 
                 (EmbeddingAuditus, QueueAuditus, "Auditus")]
async def _add_to_db_queue(spotify_track_ids: list[str]):
    LOGGER.info(f"Adding {len(spotify_track_ids)} songs to queue.")

    if not spotify_track_ids:
        return
    
    # Remove None values and duplicates
    track_ids = list(set(tid for tid in spotify_track_ids if tid is not None))

    async with get_session() as s:
        new_tracks = []
        
        for emb_type, q_type, name in QUEUE_OBJECTS:
            existing_queued = await check_multiple_in_table(
                q_type, q_type.spotify_id, track_ids, session=s
            )
            
            songs_lookup = await check_multiple_in_table(
                Song, Song.spotify_id, track_ids, session=s
            )
            
            existing_embedded = {}
            song_ids_to_check = [song.song_id for song in songs_lookup.values() if song is not None]
            if song_ids_to_check:
                embedded_lookup = await check_multiple_in_table(
                    emb_type, emb_type.song_id, song_ids_to_check, session=s
                )
                # Map back to spotify_id
                for spotify_id, song in songs_lookup.items():
                    if song and song.song_id in embedded_lookup:
                        existing_embedded[spotify_id] = embedded_lookup[song.song_id]
            
            # Add tracks that need to be queued
            for track_id in track_ids:
                if existing_queued.get(track_id):
                    LOGGER.debug(f"Song {track_id} already queued for {name}.")
                    continue
                
                if existing_embedded.get(track_id):
                    LOGGER.debug(f"Song {track_id} already embedded by {name}.")
                    continue
                
                LOGGER.debug(f"Pushing song {track_id} to {name} queue.")
                new_tracks.append(q_type(spotify_id=track_id))
        
        if new_tracks:
            LOGGER.debug(f"About to add {len(new_tracks)} queue items")
            s.add_all(new_tracks)
            # Session will be committed by context manager
        
    LOGGER.info(f"Added {len(new_tracks)} queue items.")

# TODO: Some kind of live/remastered filter? Those are effectively duplicates (in most cases).

async def queue_sp_user():  # TODO: Take userID instead of current user.
    LOGGER.info("Queueing user's Spotify library")
    liked = _get_sp_liked_tracks()
    playlist_ids = _get_sp_user_playlists()
    
    LOGGER.debug(f"Processing {len(playlist_ids)} playlists")
    playlist_tracks = [track for p in playlist_ids
                       for track in _get_sp_playlist_tracks(p)]
    
    await _add_to_db_queue(liked + playlist_tracks)

async def queue_sp_history(history: list[dict]):
    LOGGER.info(f"Queueing {len(history)} tracks from listening history to embed queue.")
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
        await s.refresh(artists)


def _get_mb_artist_tags(musicbrainz_id: str | None) -> list[str]:
    # We could also use MBID for this but on the very first artist (CAN) I tried it was already wrong.
    if not musicbrainz_id: return []

    try:
        res = mb.get_artist_by_id(artist_id, includes=["tags", "user-tags"])
        return [x["name"].lower() for x in res["artist"].get("tag-list", [])]
    except:
        # LOGGER.warning(f"Something went wrong with the request: {traceback.format_exc()}")
        return []
    
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError

@pass_session_capable
async def create_push_artist(spotify_id: str, session=None) -> Artist | None:
    artist_name = sp.artist(spotify_id)["name"]
    LOGGER.info(f"Adding {artist_name} to Artists with metadata.")

    artist_result = await session.execute(
        select(Artist).where(Artist.spotify_id == spotify_id)
    )
    if artist := artist_result.scalar_one_or_none():
        return artist

    try:
        stmt = insert(Artist).values(
            spotify_id=spotify_id, 
            artist_name=artist_name
        )
        stmt = stmt.on_conflict_do_nothing(index_elements=['spotify_id'])
        result = await session.execute(stmt)
        
        if result.rowcount > 0:
            artist_result = await session.execute(
                select(Artist).where(Artist.spotify_id == spotify_id)
            )
            artist = artist_result.scalar_one()
            
            metadata = []
            if lfm_artist := _sp_to_lastfm(spotify_id):
                LOGGER.debug(f"Adding LastFM tags to artist '{artist_name}'.")
                metadata.extend([
                    ArtistMetadata(
                        artist_id=artist.artist_id,
                        type=MetadataType.genre, 
                        value=tag.item.get_name().strip().capitalize(), 
                        source="LastFM"
                    ) for tag in lfm_artist.get_top_tags()
                ])

            if mb_tags := _get_mb_artist_tags(_sp_to_mb(spotify_id)):
                LOGGER.debug(f"Adding MusicBrainz tags to artist '{artist_name}'.")
                metadata.extend([
                    ArtistMetadata(
                        artist_id=artist.artist_id,
                        type=MetadataType.genre, 
                        value=tag, 
                        source="MusicBrainz"
                    ) for tag in mb_tags
                ])
            
            if metadata:
                session.add_all(metadata)
        else:
            artist_result = await session.execute(
                select(Artist).where(Artist.spotify_id == spotify_id)
            )
            artist = artist_result.scalar_one()
            LOGGER.debug(f"Artist '{artist_name}' already exists, using existing record.")
        
        await session.commit()
        await session.refresh(artist)
        
    except IntegrityError as e:
        await session.rollback()
        LOGGER.debug(f"Race condition detected for artist '{artist_name}', fetching existing.")
        artist_result = await session.execute(
            select(Artist).where(Artist.spotify_id == spotify_id)
        )
        artist = artist_result.scalar_one_or_none()

    LOGGER.info(f"Successfully created/retrieved artist '{artist_name}'.")
    return artist

@pass_session_capable
async def create_push_track(spotify_id: str, session=None) -> Song:
    await asyncio.sleep(0.25)  # Prevent rate-limit.
    track = sp.track(spotify_id)
    LOGGER.info(f"Adding {track['name']} to Songs with metadata.")

    if song := await check_in_table(Song, Song.spotify_id, spotify_id, session=session):
        LOGGER.info(f"Song {track['name']} already exists.")
        return song

    artists = []
    for artist_data in track["artists"]:
        if artist := await create_push_artist(artist_data["id"], session=session):
            artists.append(artist)

    try:
        song = Song(
            spotify_id=spotify_id, 
            song_name=track["name"],
            artists=artists
        )

        metadata = []
        try:
            if lfm_song := lastfm.get_track(artist=track["artists"][0]["name"], title=track["name"]):
                LOGGER.debug(f"Adding LastFM tags to track '{track['name']}'.")
                metadata.extend([
                    SongMetadata(
                        type=MetadataType.genre, 
                        value=tag.item.get_name().strip().capitalize(), 
                        source="LastFM"
                    ) for tag in lfm_song.get_top_tags()
                ])
        except Exception as e:
            LOGGER.debug(f"Failed to get LastFM data for {track['name']}: {e}")

        try:
            if mb_songs := mb.search_recordings(f'artist:"{track["artists"][0]["name"]}" AND recording:"{track["name"]}"'):
                LOGGER.debug(f"Adding MusicBrainz tags to track '{track['name']}'.")
                if mb_songs.get("recording-list"):
                    tags_data = mb.get_recording_by_id(
                        mb_songs["recording-list"][0]["id"],
                        includes=["tags", "user-tags"]
                    )
                    metadata.extend([
                        SongMetadata(
                            type=MetadataType.genre, 
                            value=tag["name"],
                            source="MusicBrainz"
                        ) for tag in tags_data["recording"].get("tag-list", [])
                    ])
        except Exception as e:
            LOGGER.debug(f"Failed to get MusicBrainz data for {track['name']}: {e}")

        song.extra_data = metadata
        session.add(song)
        await session.commit()
        await session.refresh(song)

    except IntegrityError:
        await session.rollback()
        LOGGER.debug(f"Song {track['name']} was created by another process, fetching existing.")
        song_result = await session.execute(
            select(Song).where(Song.spotify_id == spotify_id)
        )
        song = song_result.scalar_one()

    LOGGER.info(f"Successfully created/retrieved song '{song.song_name}'.")
    return song

@pass_session_capable
async def _add_song_listens(user_id: int, tracks: list[dict], session=None):
    """Add song listens with their chunks."""
    LOGGER.info(f"Adding {len(tracks)} listens to user {user_id}.")
    
    for track in tracks:
        # Check required fields
        missing_required = [required for required in ["spotify_id", "ms_played"] 
                            if required not in track]
        if missing_required:
            LOGGER.warning(f"Required fields '{"', '".join(missing_required)}' " \
                           f"missing in listen entry: {track}")
            continue
        
        # Get or create the song
        song = await create_push_track(track["spotify_id"])
        
        # Create the liten entry
        new_listen = Listen(
            user_id=user_id,
            song_id=song.song_id,
            reason_start=track.get("reason_start"), 
            reason_end=track.get("reason_end"),
            listened_at=track.get("listened_at", datetime.now()),
            ms_played=track["ms_played"]
        )
        
        session.add(new_listen)
        
        # Need to flush to get the listen_id for chunks
        await session.flush()
        
        if "chunks" in track and track["chunks"]:
            for chunk_data in track["chunks"]:
                chunk = ListenChunk(
                    listen_id=new_listen.listen_id,
                    from_ms=chunk_data["from_ms"],
                    to_ms=chunk_data["to_ms"]
                )
                session.add(chunk)
    
    try:
        await session.commit()
        LOGGER.info(f"Successfully added {len(tracks)} listen entries with chunks.")
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        await session.rollback()
        LOGGER.error(f"Error adding listens: {e}")
        raise

@pass_session_capable
async def push_sp_user_to_db(spotify_id=None, session=None) -> User:
    LOGGER.info(f"Adding user '{spotify_id}' to DB." if spotify_id else 
                "Adding current Spotify user to DB.")

    try:
        sp_user = sp.current_user()
        LOGGER.debug(f"Retrieved Spotify user: {sp_user.get('display_name', 'Unknown')} " \
                     f"({sp_user.get('id', 'Unknown ID')})")
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        LOGGER.error(f"Failed to get Spotify user: {traceback.format_exc()}")
        raise Exception("Failed to get Spotify user.")

    try:
        stmt = insert(User).values(
            spotify_id=sp_user["id"], 
            username=sp_user["display_name"]
        )
        stmt = stmt.on_conflict_do_nothing(index_elements=['spotify_id'])
        result = await session.execute(stmt)

        if result.rowcount > 0:
            LOGGER.info(f"Successfully added new user: {sp_user['display_name']} ({sp_user['id']})")
        else:
            LOGGER.debug(f"User {sp_user['display_name']} ({sp_user['id']}) already exists in database")

        user_result = await session.execute(
            select(User).where(User.spotify_id == sp_user["id"])
        )
        user = user_result.scalar_one()

        await session.commit()
        await session.refresh(user)
        
        LOGGER.info(f"User record ready: {user.username} (ID: {user.user_id})")
        return user
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        await session.rollback()
        LOGGER.error(f"Error adding user to database: {e}")
        LOGGER.debug(f"User creation error details: {traceback.format_exc()}")
        raise

@pass_session_capable
async def _get_user(spotify_id: str, session=None) -> User | None:
    result = await session.execute(select(User).where(User.spotify_id == spotify_id).limit(1))
    user = result.scalar_one_or_none()

    if user is None:
        LOGGER.info(f"User '{spotify_id}' not found, creating.")
        user = await push_sp_user_to_db(spotify_id, session=session)

    return user
        

from collections import defaultdict
START_REASON_MAP = defaultdict(lambda: "unknown", {"clickrow": "selected",
                                                   "fwdbtn": "selected",
                                                   "trackdone": "trackdone",
                                                   "backbtn": "restarted"})

END_REASON_MAP = defaultdict(lambda: "unknown", {"clickrow": "skipped",
                                                 "fwdbtn": "skipped",
                                                 "trackdone": "trackdone",
                                                 "backbtn": "restarted"})

async def add_history_listens(user_spotify_id: str, history: list[dict]):
    LOGGER.info(f"Adding {len(history)} history listens for user {user_spotify_id}")

    history = [{**listen,
                "from_history": True,
                "spotify_id": listen["spotify_track_uri"].split(":")[-1],
                "listened_at": listen["ts"],
                "reason_start": START_REASON_MAP[listen["reason_start"]],
                "reason_end": END_REASON_MAP[listen["reason_end"]],
                } for listen in history
                if listen["spotify_track_uri"] is not None and listen["spotify_episode_uri"] is None]
    LOGGER.debug(f"Processed {len(history)} history entries with mapped reasons")

    user = await _get_user(user_spotify_id)
    LOGGER.debug(f"Found user {user.user_id} for Spotify ID {user_spotify_id}")

    await _add_song_listens(user.user_id, history)
    await _add_to_db_queue([listen["spotify_id"] for listen in history])
    
    LOGGER.info(f"Successfully added {len(history)} history listens and queued songs for embedding")


SLEEP_TIME_S = 5
async def add_recent_listen_loop(user_spotify_id: str):
    LOGGER.info(f"Starting recent listen loop for user {user_spotify_id}")

    current_listen = None
    current_reason_start = "unknown"

    next_in_queue_id = None

    ms_played = 0

    listen_chunks = []
    latest_chunk_start = 0

    user = await _get_user(user_spotify_id)
    LOGGER.debug(f"Found user {user.user_id} for listen loop")

    while True:
        try:
            await asyncio.sleep(SLEEP_TIME_S)
            new_listen = sp.current_playback()
            
            try:
                # NOTE: This queue call doesn't tell us whether the q-item was deliberately
                # put in the queue orjust happens to be the next thing in the playlist/album/etc.
                q = sp.queue()["queue"]
                if len(q) >= 1:
                    if next_in_queue_id != q[0]["id"]:
                        LOGGER.debug(f"Next in queue: {next_in_queue_id}")
                        next_in_queue_id = q[0]["id"]

            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                LOGGER.warning(f"Failed to get queue: {traceback.format_exc()}")
                next_in_queue_id = None

            if not new_listen or not new_listen.get("item") or not new_listen["is_playing"]:
                LOGGER.debug("No current playback, continuing loop")
                continue

            ms_played += SLEEP_TIME_S * 1000

            if current_listen is None:
                current_listen = new_listen
                LOGGER.debug(f"Initialized listen tracking for: {current_listen['item']['name']}")
                continue

            near_start = current_listen["item"]["duration_ms"] * 0.1
            if current_listen["item"]["id"] == new_listen["item"]["id"]:
                if new_listen["progress_ms"] < current_listen["progress_ms"]:
                    LOGGER.debug(f"Song rewound from {current_listen['progress_ms']}ms " \
                                  f"to {new_listen['progress_ms']}ms")

                    listen_chunks.append({"from_ms": latest_chunk_start, 
                                          "to_ms": current_listen["progress_ms"]})
                    latest_chunk_start = new_listen["progress_ms"]

                    if new_listen["progress_ms"] < near_start:  # Restart
                        LOGGER.debug(f"Song restarted: {current_listen['item']['name']}")

                        await _add_song_listens(user.user_id, 
                                                [{"spotify_id": current_listen["item"]["id"],
                                                  "source": "live",
                                                  "ms_played": ms_played,
                                                  "reason_start": current_reason_start,
                                                  "reason_end": "restarted",
                                                  "chunks": listen_chunks}])
                        current_reason_start = "restarted"
                        listen_chunks = []
                        latest_chunk_start = 0
                elif new_listen["progress_ms"] - current_listen["progress_ms"] > SLEEP_TIME_S * 5000:
                    LOGGER.debug(f"Song fast-forwarded from {current_listen['progress_ms']}ms to {new_listen['progress_ms']}ms")

                    listen_chunks.append({"from_ms": latest_chunk_start, 
                                          "to_ms": current_listen["progress_ms"]})
                    latest_chunk_start = new_listen["progress_ms"]

                current_listen = new_listen
                continue

            # New song, process old.
            LOGGER.debug(f"Song changed from '{current_listen['item']['name']}' " \
                         f"to '{new_listen['item']['name']}'")

            near_end = current_listen["item"]["duration_ms"] * 0.75
            # TODO: Better way to model trackdone (messy with our vs spotify ms played).
            if ms_played >= near_end:  
                reason_end = "trackdone"
                new_reason_start = "trackdone"
                LOGGER.debug(f"Previous song completed naturally ({ms_played}ms >= {near_end}ms)")
            elif new_listen["item"]["id"] == next_in_queue_id:
                reason_end = "skipped"
                new_reason_start = "skipped"
                LOGGER.debug("Previous song skipped via queue")
            elif not current_listen["is_playing"]:
                reason_end = "skipped" if current_listen["is_playing"] else "paused"
                new_reason_start = "selected"
                LOGGER.debug("Previous song was paused")
            else:
                reason_end = "unknown"
                new_reason_start = "unknown"
                LOGGER.debug("Unknown reason for song change")


            if current_listen["progress_ms"] > latest_chunk_start:  # Not adding 0ms chunks.
                listen_chunks.append({"from_ms": latest_chunk_start, 
                                      "to_ms": current_listen["progress_ms"]})
            listen_data = {
                "spotify_id": current_listen["item"]["id"],
                "source": "live",
                "ms_played": ms_played,
                "reason_start": current_reason_start,
                "reason_end": reason_end,
                "chunks": listen_chunks
            }
            
            LOGGER.debug(f"Saving listen: {ms_played}ms of '{current_listen['item']['name']}' "
                        f"(start: {current_reason_start}, end: {reason_end})")
            
            await _add_song_listens(user.user_id, [listen_data])

            current_listen = new_listen
            current_reason_start = new_reason_start
            latest_chunk_start = 0
            listen_chunks = []
            latest_chunk_start = 0
            ms_played = 0 
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except ReadTimeout:
            LOGGER.warning(f"Recent listen loops got timeout on request - could be ratelimit.")
        except Exception as e:
            LOGGER.error(f"Listen loop error details: {traceback.format_exc()}")

