"""
Entrypoint. Loads config from the environment, wires up the external clients
(Spotify/MusicBrainz/LastFM/downloader), and runs four concurrent things:
  - the daily retry_pending_metadata sweep (see services.py)
  - the embedding worker processes (see embedders/)
  - the live listen-tracking loop (services.run_recent_listen_loop)
  - the download loop, feeding those embedding worker processes (see
    download.py) - includes its own periodic downloads-folder cleanup, see
    download.py's module docstring for why that's not a separate
    BackgroundScheduler job the way old/downloader.py had it.

Run as `python -m collecter.main` from src/ (relative imports require the
package context) - see docker/collecter/Dockerfile.
"""

import asyncio
import logging
import os
import sys

import pylast
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from ..db import get_session
from ..logger import setup_multiprocess_logging
from ..models import QueueAuditus, QueueJukeMIR
from . import download, embedders, services
from .clients import lastfm, musicbrainz, spotify
from .clients.downloader import SpotdlDownloaderClient
from .repository import SqlAlchemyRepository

load_dotenv()

# --- CLI flags (matches old/main.py's flag set) -----------------------------

LOG_LEVEL = logging.INFO
if "-ll" in sys.argv:
    idx = sys.argv.index("-ll") + 1
    if idx >= len(sys.argv):
        raise ValueError(
            "Expected log level value after -ll, one of ([d]ebug, [i]nfo, [w]arning, [e]rror)."
        )
    match sys.argv[idx]:
        case "debug" | "d":
            LOG_LEVEL = logging.DEBUG
        case "info" | "i":
            LOG_LEVEL = logging.INFO
        case "warning" | "w":
            LOG_LEVEL = logging.WARNING
        case "error" | "e":
            LOG_LEVEL = logging.ERROR
        case _:
            raise ValueError(
                f"Expected one of ([d]ebug, [i]nfo, [w]arning, [e]rror) for log level, "
                f"not {sys.argv[idx]}"
            )

PUSH_HISTORY = "--push-hist" in sys.argv
EMBEDDER_SELECTION = ([] if "--no-juke" in sys.argv else [QueueJukeMIR]) + (
    [] if "--no-audi" in sys.argv else [QueueAuditus]
)
assert EMBEDDER_SELECTION, (
    "Need at least 1 embedding model - don't pass both --no-juke and --no-audi."
)

setup_multiprocess_logging(console_level=LOG_LEVEL)
LOGGER = logging.getLogger(__name__)

if "--no-juke" in sys.argv:
    LOGGER.info("Not using JukeMIR.")
if "--no-audi" in sys.argv:
    LOGGER.info("Not using Auditus.")

# --- Client setup -------------------------------------------------------
#
# Same env var names as old/metadata.py, so existing .env files keep working.
# Client instances are built once here and passed down explicitly everywhere
# (resolution.py/services.py never import these modules' singletons) - that's
# what keeps those layers testable with fakes.

SPOTIFY_CLIENT = spotify.SpotifyClient(
    client_id=os.environ["SPOTIFY_CLIENT_ID"],
    client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
    redirect_uri=os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback"),
    scopes=[
        "user-library-read",
        "playlist-read-private",
        "playlist-read-collaborative",
        "user-read-currently-playing",
        "user-read-playback-state",
    ],
    cache_path=os.environ.get("SPOTIFY_CACHE_PATH", ".spotify_cache"),
)

LASTFM_CLIENT = lastfm.LastFMClient(
    api_key=os.environ["LASTFM_API_KEY"],
    api_secret=os.environ["LASTFM_API_SECRET"],
    username=os.environ["LASTFM_USERNAME"],
    password_hash=pylast.md5(os.environ["LASTFM_PW"]),
)

musicbrainz.configure(
    app_name="Suggest: Music Recommender",
    app_version="1.0",
    contact_email=os.environ["EMAIL_ADDR"],
    username=os.environ["MB_USERNAME"],
    password=os.environ["MB_PW"],
)

MUSICBRAINZ_CLIENT = musicbrainz  # module-level singleton - see clients/musicbrainz.py docstring

DOWNLOAD_DIR = os.environ.get("DOWNLOAD_DIR", "./downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

DOWNLOADER_CLIENT = SpotdlDownloaderClient(
    client_id=os.environ["SPOTIFY_CLIENT_ID"],
    client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
    refresh_token=os.environ["SPOTIFY_REFRESH_TOKEN"],
    download_dir=DOWNLOAD_DIR,
)


def _run_retry_sweep():
    """Sync wrapper - BackgroundScheduler jobs are plain callables, not coroutines."""

    async def _sweep():
        async with get_session() as session:
            repo = SqlAlchemyRepository(session)
            await services.retry_pending_metadata(
                repo, SPOTIFY_CLIENT, MUSICBRAINZ_CLIENT, LASTFM_CLIENT
            )

    asyncio.run(_sweep())


async def main():
    LOGGER.info("=== Collecter starting ===")

    scheduler = BackgroundScheduler()
    scheduler.add_job(_run_retry_sweep, "interval", days=1)
    scheduler.start()

    song_queues = embedders.start_processes(EMBEDDER_SELECTION)

    try:
        async with get_session() as session:
            repo = SqlAlchemyRepository(session)

            user = await services.push_user(None, repo, SPOTIFY_CLIENT)

            if PUSH_HISTORY:
                history_folder = os.environ.get("SPOTIFY_HISTORY_FOLDER")
                if not history_folder:
                    LOGGER.warning(
                        "--push-hist given but SPOTIFY_HISTORY_FOLDER is not set - skipping."
                    )
                else:
                    LOGGER.info(f"Pushing history from {history_folder} to the embed queue.")
                    await services.queue_history_folder(history_folder, repo)

            await asyncio.gather(
                services.run_recent_listen_loop(
                    user.spotify_id, repo, SPOTIFY_CLIENT, MUSICBRAINZ_CLIENT, LASTFM_CLIENT
                ),
                download.download_loop(song_queues, repo, DOWNLOADER_CLIENT, DOWNLOAD_DIR),
            )
    except KeyboardInterrupt:
        LOGGER.info("Shutting down.")
    finally:
        scheduler.shutdown()
        embedders.end_processes()


if __name__ == "__main__":
    asyncio.run(main())
