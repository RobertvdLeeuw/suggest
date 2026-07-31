"""
Purpose & scope
---------------
Orchestration only. Glues resolution.py + mapping.py + repository.py (and,
for live listens, listen_tracking.py) together. This is what downloader.py
and main.py actually call.

Rules for this file:
  - Always check the repository before resolving. Resolution costs external
    API calls (and rate-limit budget); an already-known Artist/Song must
    short-circuit before touching Spotify/MusicBrainz/LastFM again. The
    order in every push_* function below is: repo lookup -> (miss) resolve
    -> mapping -> repo create. Never resolve-then-check.
  - This file is expected to be the least-tested layer by design - its two
    ingredients (resolution, repository) are each fully covered on their
    own, so services.py mostly needs a handful of orchestration-ordering
    tests (e.g. "doesn't call resolve when repo already has it") rather than
    exhaustive coverage.

On resolution.py's `unavailable` (see its module docstring): every push_*
function that creates a new Artist/Song writes the row with whatever tags it
got AND records any unavailable sources via repo.mark_*_metadata_pending -
never blocks the write entirely. Blocking would mean a LastFM hiccup stops
new listens from being recorded at all, which is worse than a temporarily
under-tagged row. retry_pending_metadata is what closes the gap later, on
its own schedule (see main.py) - not by hoping something re-pushes the same
artist/song again.
"""

import asyncio
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from ..models import Artist, Listen, Song, User
from . import mapping, resolution
from .clients import LastFMClientProtocol, MusicBrainzClientProtocol, SpotifyClientProtocol
from .listen_tracking import TrackingState, process_playback_tick
from .repository import Repository

# Spotify's own reason codes -> the StartEndReason vocabulary listen_tracking.py/
# models.py use. Ported from old/metadata.py's START_REASON_MAP/END_REASON_MAP -
# only used for extended-history imports; the live polling loop (listen_tracking.py)
# classifies reasons itself, it never sees these raw Spotify codes.
_HISTORY_START_REASON_MAP = defaultdict(
    lambda: "unknown",
    {
        "clickrow": "selected",
        "fwdbtn": "selected",
        "trackdone": "trackdone",
        "backbtn": "restarted",
    },
)
_HISTORY_END_REASON_MAP = defaultdict(
    lambda: "unknown",
    {"clickrow": "skipped", "fwdbtn": "skipped", "trackdone": "trackdone", "backbtn": "restarted"},
)


async def push_artist(
    spotify_id: str,
    repo: Repository,
    spotify: SpotifyClientProtocol,
    musicbrainz: MusicBrainzClientProtocol,
    lastfm: LastFMClientProtocol,
) -> Artist:
    """Repo lookup first; only resolves against external APIs on a miss."""
    artist = await repo.get_artist_by_spotify_id(spotify_id)
    if artist is not None:
        return artist

    resolved = await resolution.resolve_artist(spotify_id, spotify, musicbrainz, lastfm)
    artist_orm = mapping.artist_to_orm(resolved)
    artist = await repo.create_artist(artist_orm, artist_orm.extra_data)

    if resolved.unavailable:
        await repo.mark_artist_metadata_pending(artist.artist_id, resolved.unavailable)

    return artist


async def push_track(
    spotify_id: str,
    repo: Repository,
    spotify: SpotifyClientProtocol,
    musicbrainz: MusicBrainzClientProtocol,
    lastfm: LastFMClientProtocol,
) -> Song:
    """Repo lookup first; only resolves against external APIs on a miss.
    Resolves/pushes each artist on the track via push_artist."""
    song = await repo.get_song_by_spotify_id(spotify_id)
    if song is not None:
        return song

    resolved = await resolution.resolve_track(spotify_id, spotify, musicbrainz, lastfm)
    artists = [
        await push_artist(artist.spotify_id, repo, spotify, musicbrainz, lastfm)
        for artist in resolved.artists
    ]

    song_orm = mapping.track_to_orm(resolved, artists)
    song = await repo.create_song(song_orm, artists, song_orm.extra_data)

    if resolved.unavailable:
        await repo.mark_song_metadata_pending(song.song_id, resolved.unavailable)

    return song


async def push_user(
    spotify_id: str | None,
    repo: Repository,
    spotify: SpotifyClientProtocol,
) -> User:
    """spotify_id=None means 'the currently authenticated user'."""
    sp_user = await spotify.current_user()
    return await repo.get_or_create_user(spotify_id or sp_user["id"], sp_user["display_name"])


async def add_song_listens(
    user_id: int,
    tracks: list[dict],
    repo: Repository,
    spotify: SpotifyClientProtocol,
    musicbrainz: MusicBrainzClientProtocol,
    lastfm: LastFMClientProtocol,
) -> None:
    """Ensures each track is pushed (push_track) before recording the listen."""
    for track in tracks:
        missing = [f for f in ("spotify_id", "ms_played") if f not in track]
        if missing:
            continue  # malformed entry - old code logged and skipped, same here

        song = await push_track(track["spotify_id"], repo, spotify, musicbrainz, lastfm)
        await repo.add_listen(
            user_id,
            song.song_id,
            {
                "ms_played": track["ms_played"],
                "reason_start": track.get("reason_start"),
                "reason_end": track.get("reason_end"),
                "listened_at": track.get("listened_at"),
                "chunks": track.get("chunks", []),
                "from_history": track.get("from_history", False),
            },
        )


async def add_history_listens(
    user_spotify_id: str,
    history: list[dict],
    repo: Repository,
    spotify: SpotifyClientProtocol,
    musicbrainz: MusicBrainzClientProtocol,
    lastfm: LastFMClientProtocol,
) -> None:
    """Maps raw Spotify extended-history entries (reason codes etc.) and delegates
    to add_song_listens."""
    mapped = [
        {
            **listen,
            "from_history": True,
            "spotify_id": listen["spotify_track_uri"].split(":")[-1],
            "listened_at": listen["ts"],
            "reason_start": _HISTORY_START_REASON_MAP[listen["reason_start"]],
            "reason_end": _HISTORY_END_REASON_MAP[listen["reason_end"]],
        }
        for listen in history
        if listen.get("spotify_track_uri") is not None and listen.get("spotify_episode_uri") is None
    ]

    user = await push_user(user_spotify_id, repo, spotify)
    await add_song_listens(user.user_id, mapped, repo, spotify, musicbrainz, lastfm)
    await queue_new_tracks([listen["spotify_id"] for listen in mapped], repo)


async def queue_new_tracks(spotify_track_ids: list[str], repo: Repository) -> None:
    ids = [tid for tid in spotify_track_ids if tid is not None]
    await repo.enqueue_tracks(ids)


async def queue_sp_library(
    repo: Repository,
    spotify: SpotifyClientProtocol,
) -> None:
    """Queues every track in the current user's liked songs and playlists for
    embedding. Ports old/metadata.py's queue_sp_user - unlike that version this
    takes an explicit SpotifyClientProtocol rather than reaching for a
    module-level singleton, same convention as everything else in this file.
    Always the currently authenticated user, same as push_user(None, ...) -
    there's no per-user library on someone else's Spotify account to queue."""
    liked = await spotify.current_user_saved_tracks()
    liked_ids = [
        t["track"]["id"] for t in liked["items"] if t.get("track") and t["track"].get("id")
    ]

    playlists = await spotify.current_user_playlists()
    playlist_ids = [p["id"] for p in playlists["items"] if p.get("id")]

    playlist_track_ids: list[str] = []
    for playlist_id in playlist_ids:
        playlist = await spotify.playlist(playlist_id)
        playlist_track_ids.extend(
            t["track"]["id"]
            for t in playlist["tracks"]["items"]
            if t.get("track") and t["track"].get("id")
        )

    await queue_new_tracks(liked_ids + playlist_track_ids, repo)


async def queue_history_folder(folder: str, repo: Repository) -> None:
    """Reads every .json file in `folder` (a Spotify Extended Streaming History
    export) and queues its tracks for embedding. The one function in this file
    that does filesystem I/O rather than DB/network - that's fine, it's still
    just gathering ids to hand to queue_new_tracks, not a second responsibility."""
    import json

    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(folder, filename)) as f:
            chunk = json.load(f)

        track_ids = [
            listen["spotify_track_uri"].split(":")[-1]
            for listen in chunk
            if listen.get("spotify_track_uri")
        ]
        await queue_new_tracks(track_ids, repo)


async def queue_similar_artists(
    repo: Repository,
    spotify: SpotifyClientProtocol,
    lastfm: LastFMClientProtocol,
) -> None:
    """Picks known artists at random, finds similar ones via resolution.get_similar_artists,
    and queues their tracks."""
    artists = await repo.get_random_artists(10)

    similar_ids: list[str] = []
    for artist in artists:
        similar_ids.extend(await resolution.get_similar_artists(artist.spotify_id, spotify, lastfm))

    tracks: list[str] = []
    for artist_id in similar_ids:
        albums = await spotify.artist_albums(artist_id)
        for album in albums["items"]:
            album_tracks = await spotify.album_tracks(album["id"])
            tracks.extend(t["id"] for t in album_tracks["items"])

    await queue_new_tracks(tracks, repo)


async def retry_pending_metadata(
    repo: Repository,
    spotify: SpotifyClientProtocol,
    musicbrainz: MusicBrainzClientProtocol,
    lastfm: LastFMClientProtocol,
    older_than: timedelta = timedelta(days=1),
) -> None:
    """Re-resolves any artist/song with a source still marked pending (UNAVAILABLE
    on a previous attempt) for at least `older_than`, and clears whichever sources
    succeed this time. Meant to run on a schedule (see main.py) - the schedule's own
    interval is the retry backoff, so there's no per-row timestamp math here: a
    source that fails again just stays in the pending table, to be picked up by the
    next scheduled run.

    Re-creating an artist/song that already exists is safe - repository.create_artist/
    create_song both go through get_or_create/get_or_create_many, so this only adds
    whatever tags are newly available, it never duplicates or overwrites anything."""
    cutoff = datetime.now(timezone.utc) - older_than

    for artist in await repo.get_stale_pending_artists(cutoff):
        resolved = await resolution.resolve_artist(artist.spotify_id, spotify, musicbrainz, lastfm)
        artist_orm = mapping.artist_to_orm(resolved)
        await repo.create_artist(artist_orm, artist_orm.extra_data)

        newly_resolved = resolution.KNOWN_SOURCES - resolved.unavailable
        await repo.clear_artist_metadata_pending(artist.artist_id, newly_resolved)

    for song in await repo.get_stale_pending_songs(cutoff):
        resolved = await resolution.resolve_track(song.spotify_id, spotify, musicbrainz, lastfm)
        artists = [
            await push_artist(artist.spotify_id, repo, spotify, musicbrainz, lastfm)
            for artist in resolved.artists
        ]
        song_orm = mapping.track_to_orm(resolved, artists)
        await repo.create_song(song_orm, artists, song_orm.extra_data)

        newly_resolved = resolution.KNOWN_SOURCES - resolved.unavailable
        await repo.clear_song_metadata_pending(song.song_id, newly_resolved)


async def run_recent_listen_loop(
    user_spotify_id: str,
    repo: Repository,
    spotify: SpotifyClientProtocol,
    musicbrainz: MusicBrainzClientProtocol,
    lastfm: LastFMClientProtocol,
    sleep_time_s: int = 5,
) -> None:
    """The actual polling loop: sleeps, calls spotify.current_playback(), advances
    listen_tracking.process_playback_tick, and persists any resulting ListenEvent via
    repo.add_listen. Runs forever - intended to be one of the tasks in main.py's
    asyncio.gather.

    Every newly-encountered song (by id, on start or change - not on resume/restart)
    is pushed to the embedding queues via repo.enqueue_tracks, independent of whether
    its listen has concluded yet. Concluded listens are recorded even for songs not
    yet in the DB - push_track resolves/creates them first, same repo-lookup-before-
    resolve rule as every other push_* function here."""
    user = await push_user(user_spotify_id, repo, spotify)
    state = TrackingState()

    while True:
        await asyncio.sleep(sleep_time_s)
        new_snapshot = await spotify.current_playback()

        try:
            queue = await spotify.queue()
            next_in_queue_id = queue["queue"][0]["id"] if queue["queue"] else None
        except Exception:
            next_in_queue_id = None

        old_id = state.current_listen["item"]["id"] if state.current_listen else None

        state, event = process_playback_tick(state, new_snapshot, next_in_queue_id, sleep_time_s)

        new_id = state.current_listen["item"]["id"] if state.current_listen else None
        if new_id and new_id != old_id:
            await repo.enqueue_tracks([new_id])

        if event:
            song = await push_track(event.spotify_id, repo, spotify, musicbrainz, lastfm)
            await repo.add_listen(
                user.user_id,
                song.song_id,
                {
                    "ms_played": event.ms_played,
                    "reason_start": event.reason_start,
                    "reason_end": event.reason_end,
                    "chunks": [{"from_ms": c.from_ms, "to_ms": c.to_ms} for c in event.chunks],
                },
            )


if __name__ == "__main__":
    """Standalone entrypoint for queue_sp_library - run as
    `python -m collecter.services` (same package-context requirement as
    main.py: relative imports below need it run as a module, not a script).
    Only needs a repo session + Spotify client, unlike main.py's full
    Spotify/MusicBrainz/LastFM setup - LastFM/MusicBrainz are irrelevant to
    queuing a library for embedding."""
    import logging
    import os

    from dotenv import load_dotenv

    from db import get_session

    from .clients import spotify
    from .repository import SqlAlchemyRepository

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    async def _run():
        spotify_client = spotify.SpotifyClient(
            client_id=os.environ["SPOTIFY_CLIENT_ID"],
            client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
            redirect_uri=os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback"),
            scopes=["user-library-read", "playlist-read-private", "playlist-read-collaborative"],
            cache_path=os.environ.get("SPOTIFY_CACHE_PATH", ".spotify_cache"),
        )

        async with get_session() as session:
            repo = SqlAlchemyRepository(session)
            await queue_sp_library(repo, spotify_client)

    asyncio.run(_run())
