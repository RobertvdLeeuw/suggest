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

from models import Artist, Song, User, Listen

from .clients import SpotifyClientProtocol, MusicBrainzClientProtocol, LastFMClientProtocol
from .repository import Repository
from .listen_tracking import TrackingState


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


async def queue_new_tracks(spotify_track_ids: list[str], repo: Repository) -> None:
    ...


async def queue_history_folder(folder: str, repo: Repository) -> None:
    ...


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
    sleep_time_s: int = 5,
) -> None:
    """The actual polling loop: sleeps, calls spotify.current_playback(), advances
    listen_tracking.process_playback_tick, and persists any resulting ListenEvent via
    repo.add_listen. Runs forever - intended to be one of the tasks in main.py's
    asyncio.gather."""
    ...
