from logger import LOGGER
import traceback

LOGGER.info("Starting imports...")

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from apscheduler.schedulers.asyncio import AsyncIOScheduler

import os

from embedders import start_processes, end_processes
from downloader import start_download_loop, clean_downloads, _download


async def main():
    LOGGER.info("=== Audio Processing Service Starting ===")
    LOGGER.info(f"Python PID: {os.getpid()}")
    LOGGER.debug(f"Current working directory: {os.getcwd()}")
    
    try:
        LOGGER.info("Starting embedding worker processes...")

        song_queues = start_processes()
        
        scheduler = AsyncIOScheduler()
        scheduler.add_job(clean_downloads,'interval', hours=1, args=(song_queues,))
        scheduler.start()

        await _download("https://open.spotify.com/track/3dzCClyQ3qKx2o3CLIx02r?si=0c3722a5d8bd4e61")  # Both a test AND a shitpost.
        await start_download_loop(song_queues)
    except Exception as e:
        LOGGER.error(f"Main loop error: {traceback.format_exc()}")
    finally:
        scheduler.shutdown()
        
        end_processes()

if __name__ == "__main__":
    asyncio.run(main())
