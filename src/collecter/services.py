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
"""

import asyncio

from models import Artist, Listen, Song, User

from .clients import LastFMClientProtocol, MusicBrainzClientProtocol, SpotifyClientProtocol
from .listen_tracking import TrackingState, process_playback_tick
from .repository import Repository


async def push_artist(
    spotify_id: str,
    repo: Repository,
    spotify: SpotifyClientProtocol,
    musicbrainz: MusicBrainzClientProtocol,
    lastfm: LastFMClientProtocol,
) -> Artist:
    """Repo lookup first; only resolves against external APIs on a miss."""
    ...


async def push_track(
    spotify_id: str,
    repo: Repository,
    spotify: SpotifyClientProtocol,
    musicbrainz: MusicBrainzClientProtocol,
    lastfm: LastFMClientProtocol,
) -> Song:
    """Repo lookup first; only resolves against external APIs on a miss.
    Resolves/pushes each artist on the track via push_artist."""
    ...


async def push_user(
    spotify_id: str | None,
    repo: Repository,
    spotify: SpotifyClientProtocol,
) -> User:
    """spotify_id=None means 'the currently authenticated user'."""
    ...


async def add_song_listens(
    user_id: int,
    tracks: list[dict],
    repo: Repository,
    spotify: SpotifyClientProtocol,
    musicbrainz: MusicBrainzClientProtocol,
    lastfm: LastFMClientProtocol,
) -> None:
    """Ensures each track is pushed (push_track) before recording the listen."""
    ...


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
    ...


async def queue_new_tracks(spotify_track_ids: list[str], repo: Repository) -> None: ...


async def queue_history_folder(folder: str, repo: Repository) -> None: ...


async def queue_similar_artists(
    repo: Repository,
    spotify: SpotifyClientProtocol,
    lastfm: LastFMClientProtocol,
) -> None:
    """Picks known artists at random, finds similar ones via resolution.get_similar_artists,
    and queues their tracks."""
    ...


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
