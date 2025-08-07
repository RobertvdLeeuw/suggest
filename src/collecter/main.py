import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import sys
import os

if "-t" in sys.argv or "--test" in sys.argv:
    os.environ["TEST_MODE"] = "true"


import logging
from logger import setup_multiprocess_logging

log_level = logging.INFO
if "-ll" in sys.argv:
    idx = sys.argv.index("-ll") + 1
    if idx >= len(sys.argv): raise ValueError("Expected log level value after -ll, one of ([d]ebug, [i]nfo, [w]arning, [e]rror).")

    match sys.argv[idx]:
        case "debug" | "d": log_level = logging.DEBUG
        case "info" | "i": log_level = logging.INFO
        case "warning" | "w": log_level = logging.WARNING
        case "error" | "e": log_level = logging.ERROR
        case _: raise ValueError(f"Expected one of ([d]ebug, [i]nfo, [w]arning, [e]rror) for log level, not {sys.argv[idx]}")
setup_multiprocess_logging(console_level=log_level)

LOGGER = logging.getLogger(__name__)
import traceback

if os.getenv("TEST_MODE"):
    LOGGER.info("Test mode initiated, using test DB and mocks.")

LOGGER.info("Starting imports...")

from db import setup, get_session
import traceback

import asyncio
import threading
from apscheduler.schedulers.background import BackgroundScheduler

from collecter.embedders import start_processes, end_processes
from collecter.downloader import start_download_loop, clean_downloads
from collecter.metadata import queue_sp_user, queue_sp_history, _get_sp_album_tracks, _add_to_db_queue, add_recent_listen_loop, push_sp_user_to_db, refresh_spotipy


async def main():
    LOGGER.info("=== Audio Processing Service Starting ===")
    LOGGER.info(f"Python PID: {os.getpid()}")
    LOGGER.debug(f"Current working directory: {os.getcwd()}")

    await setup()
    user = await push_sp_user_to_db()

    if "--push-hist" in sys.argv:
        LOGGER.info("Pushing history to DB.")
        await queue_sp_history()

    scheduler = None
    try:
        LOGGER.info("Starting embedding worker processes...")

        song_queues = start_processes()
        
        scheduler = BackgroundScheduler()
        scheduler.add_job(clean_downloads, 'interval', minutes=1, args=(song_queues,))
        scheduler.add_job(refresh_spotipy, 'interval', minutes=55)  # Race issues?
        scheduler.start()

        # await _add_to_db_queue(_get_sp_album_tracks("2zQeigA4bFAlTqQqBiVe6Y"))

        await asyncio.gather(start_download_loop(song_queues),
                             # queue_sp_user(),
                             add_recent_listen_loop(user.spotify_id))
    except Exception as e:
        LOGGER.error(f"Main loop error: {traceback.format_exc()}")
    finally:
        if scheduler:
            scheduler.shutdown()
        
        end_processes()

if __name__ == "__main__":
    mp.freeze_support()
    asyncio.run(main())

