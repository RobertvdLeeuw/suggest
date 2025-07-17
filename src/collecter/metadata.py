from logger import LOGGER

import traceback
import os

from dotenv import load_dotenv
load_dotenv()

import pylast
from pylast import Tag, Artist, Track
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from models import EmbeddingJukeMIR, EmbeddingAuditus, Artist

SCOPES = ['user-library-read',
          'playlist-read-private',
          'playlist-read-collaborative',
          'user-read-currently-playing',
          'user-read-playback-state']

sp_oauth = SpotifyOAuth(
    client_id=os.environ["SPOTIFY_CLIENT_ID"],
    client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
    redirect_uri="http://127.0.0.1:8080/callback",
    scope=SCOPES
)


if os.path.exists(".cache"):
    os.remove(".cache")

token_info = sp_oauth.refresh_access_token(os.environ["SPOTIFY_REFRESH_TOKEN"])
sp = spotipy.Spotify(auth=token_info['access_token'])

def get_spotipy():
    return sp

import musicbrainzngs as mb
mb.set_useragent("Suggest: Music Recommender", "1.0", contact=os.environ["EMAIL_ADDR"])
mb.set_rate_limit()
mb.auth(os.environ["MB_USERNAME"], os.environ["MB_PW"])

from models import (
        QueueJukeMIR, QueueAuditus,
        Song, SongMetadata, 
        Artist, ArtistMetadata, MetadataType, 
        ListenChunk, Listen
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

    res = mb.search_recordings(f'artist:"{artist_name}" AND recording:"{top_song}"')
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
        result = await session.execute(select(table).where(col == value).limit(1))
        # result = await session.execute(select(exists().where(col == value)))
        item = result.scalar_one_or_none()
        LOGGER.debug(f"Checking {table.__name__} for {col.name}={value}: {'Found' if item else 'Not found'}")

        return item  # TODO: Does this just grab first for multiple results?

QUEUE_OBJECTS = [(EmbeddingJukeMIR, QueueJukeMIR, "JukeMIR"), 
                 (EmbeddingAuditus, QueueAuditus, "Auditus")]
async def _add_to_db_queue(spotify_track_ids: list[str]):
    LOGGER.debug(f"Adding {len(spotify_track_ids)} songs to queue.")

    new_tracks = []
    async with get_session() as s:
        for track_id in set(spotify_track_ids): 
            for emb_type, q_type, name in QUEUE_OBJECTS:
                if await check_in_table(q_type, q_type.spotify_id, track_id, s):
                    LOGGER.debug(f"Song {track_id} already queued for {name}.")
                    continue

                if song := await check_in_table(Song, Song.spotify_id, track_id, s):
                    if await check_in_table(emb_type, emb_type.song_id, song.song_id, s):
                        LOGGER.debug(f"Song {track_id} already embedded by {name}.")
                        continue

                LOGGER.debug(f"Pushing song {track_id} to {name} queue.")
                new_tracks.append(q_type(spotify_id=track_id))

        if new_tracks:
            s.add_all(new_tracks)
            await s.commit()
    LOGGER.info(f"Added {len(new_tracks)} queue items.")

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

async def create_push_artist(spotify_id: str) -> Artist:
    artist_name = sp.artist(spotify_id)["name"]
    LOGGER.info(f"Adding {artist_name} to Artists with metadata.")

    async with get_session() as s:
        try:
            stmt = insert(Artist).values(
                spotify_id=spotify_id, 
                artist_name=artist_name
            )
            stmt = stmt.on_conflict_do_nothing(index_elements=['spotify_id'])
            result = await s.execute(stmt)
            
            if result.rowcount > 0:
                artist_result = await s.execute(
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
                    s.add_all(metadata)
            else:
                artist_result = await s.execute(
                    select(Artist).where(Artist.spotify_id == spotify_id)
                )
                artist = artist_result.scalar_one()
                LOGGER.debug(f"Artist '{artist_name}' already exists, using existing record.")
            
            await s.commit()
            await s.refresh(artist)
            
        except IntegrityError as e:
            await s.rollback()
            LOGGER.debug(f"Race condition detected for artist '{artist_name}', fetching existing.")
            artist_result = await s.execute(
                select(Artist).where(Artist.spotify_id == spotify_id)
            )
            artist = artist_result.scalar_one()

    LOGGER.info(f"Successfully created/retrieved artist '{artist_name}'.")
    return artist

async def push_track_metadata(spotify_id: str) -> Song:
    track = sp.track(spotify_id)
    LOGGER.info(f"Adding {track['name']} to Songs with metadata.")

    async with get_session() as s:
        if song := await check_in_table(Song, Song.spotify_id, spotify_id, s):
            LOGGER.info(f"Song {track['name']} already exists.")
            return song

        artists = []
        for artist_data in track["artists"]:
            artist = await create_push_artist(artist_data["id"])
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
            s.add(song)
            await s.commit()
            await s.refresh(song)

        except IntegrityError:
            await s.rollback()
            LOGGER.debug(f"Song {track['name']} was created by another process, fetching existing.")
            song_result = await s.execute(
                select(Song).where(Song.spotify_id == spotify_id)
            )
            song = song_result.scalar_one()

    LOGGER.info(f"Successfully created/retrieved song '{song.song_name}'.")
    return song

async def _add_to_db_queue(spotify_track_ids: list[str]):
    LOGGER.info(f"Adding {len(spotify_track_ids)} songs to queue.")

    new_tracks = []
    async with get_session() as s:
        for track_id in set(spotify_track_ids): 
            for emb_type, q_type, name in QUEUE_OBJECTS:
                if await check_in_table(q_type, q_type.spotify_id, track_id, s):
                    LOGGER.debug(f"Song {track_id} already queued for {name}.")
                    continue

                if song := await check_in_table(Song, Song.spotify_id, track_id, s):
                    if await check_in_table(emb_type, emb_type.song_id, song.song_id, s):
                        LOGGER.debug(f"Song {track_id} already embedded by {name}.")
                        continue

                LOGGER.debug(f"Pushing song {track_id} to {name} queue.")
                new_tracks.append(q_type(spotify_id=track_id))

        if new_tracks:
            s.execute(new_tracks, multi=True)
            await s.commit()
    LOGGER.info(f"Added {len(new_tracks)} queue items.")



async def add_song_listens(user_id: int, tracks: list[dict]):
    """Add song listens with their chunks."""
    LOGGER.info(f"Adding {len(tracks)} listens to user {user_id}.")
    
    async with get_session() as s:
        for track in tracks:
            # Check required fields
            missing_required = [required for required in ["spotify_id", "ms_played"] 
                                if required not in track]
            if missing_required:
                LOGGER.warning(f"Required fields '{"', '".join(missing_required)}' " \
                               f"missing in listen entry: {track}")
                continue
            
            # Get or create the song
            song = await push_track_metadata(track["spotify_id"])
            
            # Create the liten entry
            new_listen = Listen(
                user_id=user_id,
                song_id=song.song_id,
                reason_start=track.get("reason_start"), 
                reason_end=track.get("reason_end"),
                listened_at=track.get("listened_at", datetime.now()),
                ms_played=track["ms_played"]
            )
            
            s.add(new_listen)
            
            # Need to flush to get the listen_id for chunks
            await s.flush()
            
            if "chunks" in track and track["chunks"]:
                for chunk_data in track["chunks"]:
                    chunk = ListenChunk(
                        listen_id=new_listen.listen_id,
                        from_ms=chunk_data["from_ms"],
                        to_ms=chunk_data["to_ms"]
                    )
                    s.add(chunk)
        
        try:
            await s.commit()
            LOGGER.info(f"Successfully added {len(tracks)} listen entries with chunks.")
        except Exception as e:
            await s.rollback()
            LOGGER.error(f"Error adding listens: {e}")
            raise

async def _get_user(spotify_id: str) -> User | None:
    async with get_session() as s:
        result = await session.execute(select(User).where(User.spotify_id == spotify_id).limit(1))
        item = result.scalar_one_or_none()
    
    if item is None:
        LOGGER.warning(f"User '{spotify_id}' not found.")

    return item
        

async def add_history_listens(user_spotify_id: str, history: list[dict]):
    history = [{**listen,
                "source": "history",
                "spotify_id": listen["spotify_track_uri"].split(":")[-1],
                "listened_at": listen["ts"]
                } for listen in history]

    user = await _get_user(user_spotify_id)
    await _add_song_listens(user.user_id, history)
    await _add_to_db_queue([listen["spotify_id"] for listen in history])


SLEEP_TIME_S = 2
async def add_recent_listen_loop(user_spotify_id: str):
    current_listen = None
    current_reason_start = "unknown"

    next_in_queue_id = None

    ms_played = 0

    listen_chunks = []
    latest_chunk_start = 0

    user = await _get_user(user_spotify_id)

    while True:
        asyncio.sleep(SLEEP_TIME_S)
        ms_played += SLEEP_TIME_S * 1000
        new_listen = sp.current_playback()
        
        near_start = current_listen["item"]["duration_ms"] * 0.1
        if current_listen["item"]["id"] == new_listen["item"]["id"]:
            if new_listen["progress_ms"] <= current_listen["progress_ms"]):
                listen_chunks.append({"from_ms": latest_chunk_start, 
                                      "to_ms": current_listen["progress_ms"]})
                latest_chunk_start = new_listen["progress_ms"]

                if new_listen["progress_ms"] < near_start:  # Restart
                    await _add_song_listens(user_id, [{"spotify_id": current_listen["item"]["id"],
                                                       "source": "live",
                                                       "ms_played": ms_played,
                                                       "reason_start": current_reason_start,
                                                       "reason_end": "restarted"
                                                       "chunks": listen_chunks}])
                    current_reason_start = "restarted"
                    listen_chunks = []
                    latest_chunk_start = 0
            elif new_listen["progress_ms"] - current_listen["progress_ms"] > SLEEP_TIME_S * 2:
                listen_chunks.append({"from_ms": latest_chunk_start, 
                                      "to_ms": current_listen["progress_ms"]})
                latest_chunk_start = new_listen["progress_ms"]

            current_listen = new_listen
            continue

        # New song, process old.
        near_end = current_listen["item"]["duration_ms"] * 0.75
        # TODO: Better way to model trackdone (messy with our vs spotify ms played).
        if ms_played >= near_end:  
            reason_end = "trackdone"
            new_reason_start = "trackdone"
        elif new_listen["item"]["id"] == next_in_queue_id:
            reason_end = "skipped"
            new_reason_start = "skipped"
        elif not current_listen["is_playing"]:
            reason_end = "skipped" if current_listen["is_playing"] else "paused"
            new_reason_start = "selected"
        else:
            reason_end = "unknown"
            new_reason_start = "unknown"

        q = sp.queue()["queue"]
        if len(q) >= 1:
            next_in_queue_id = q[0]["id"]


        listen_chunks.append({"from_ms": latest_chunk_start, 
                              "to_ms": current_listen["progress_ms"]})
        await _add_song_listens(user.user_id, [{"spotify_id": current_listen["item"]["id"],
                                                "source": "live",
                                                "ms_played": ms_played,
                                                "reason_start": current_reason_start,
                                                "reason_end": reason_end,
                                                "chunks": listen_chunks}])

        current_listen = new_listen
        current_reason_start = new_reason_start
        latest_chunk_start = 0
        listen_chunks = []
        latest_chunk_start = 0
        ms_played = 0 

