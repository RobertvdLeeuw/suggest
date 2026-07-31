"""
Entrypoint - currently just wires up the scheduled retry_pending_metadata sweep
(see services.py). Everything else from old/main.py (download loop, embedding
processes, recent-listen loop, CLI flags) still needs porting in here.
"""

import asyncio
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from db import get_session
from logger import setup_multiprocess_logging

from . import services
from .clients import lastfm, musicbrainz, spotify

setup_multiprocess_logging(console_level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# TODO: real config (env vars/CLI flags, matching old/main.py's -ll, --push-hist,
# --no-juke, --no-audi handling) instead of these placeholders.
LASTFM_CLIENT: lastfm.LastFMClient
SPOTIFY_CLIENT: spotify.SpotifyClient


def _run_retry_sweep():
    """Sync wrapper - BackgroundScheduler jobs are plain callables, not coroutines."""

    async def _sweep():
        async with get_session() as session:
            from .repository import SqlAlchemyRepository

            repo = SqlAlchemyRepository(session)
            await services.retry_pending_metadata(repo, SPOTIFY_CLIENT, musicbrainz, LASTFM_CLIENT)

    asyncio.run(_sweep())


async def main():
    LOGGER.info("=== Collecter starting ===")

    scheduler = BackgroundScheduler()
    scheduler.add_job(_run_retry_sweep, "interval", days=1)
    scheduler.start()

    try:
        # TODO: download loop, embedding processes, run_recent_listen_loop,
        # everything else old/main.py currently does.
        await asyncio.Event().wait()  # placeholder - keeps the process alive
    finally:
        scheduler.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
