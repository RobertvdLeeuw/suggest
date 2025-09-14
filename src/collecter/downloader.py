import traceback
import logging
LOGGER = logging.getLogger(__name__)

import os
from pathlib import Path

import asyncio
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select, delete
from db import get_session

from spotdl.types.options import DownloaderOptions

from collecter.embedders import SongQueue, QUEUE_MAX_LEN
from collecter.metadata import (
    simple_queue_new_music, 
    create_push_track,
    _get_sp_album_tracks,
    _add_to_db_queue,
    get_spotipy
)
from spotdl.providers.audio.base import AudioProviderError
from spotdl.download.downloader import DownloaderError

DOWNLOAD_LOC = None
Spotdl = None


def spotdl_lazy_load():
    global Spotdl, LookupError

    if Spotdl is not None:
        return

    if os.getenv("TEST_MODE"):
        LOGGER.debug("Initializing mock Spotdl client.")
        from tests.mocks.embedders import Spotdl_fake as _Spotdl
    else:
        LOGGER.debug("Initializing Spotdl client.")
        from spotdl import Spotdl as _Spotdl

    Spotdl = _Spotdl

CURRENTLY_DOWNLOADING = set()
async def _download(spotify_id: str, song_queue: SongQueue):
    global DOWNLOAD_LOC
    DOWNLOAD_LOC = "./mock_downloads" if os.getenv("TEST_MODE") else "./downloads"

    if spotify_id in CURRENTLY_DOWNLOADING:
        return

    for fpath, sp_id in song_queue.peek_all():
        if spotify_id == sp_id:
            LOGGER.debug(f"Song already downloaded: {fpath}.")
            return

    LOGGER.info(f"Starting download for: {spotify_id}")
    CURRENTLY_DOWNLOADING.add(spotify_id)

    try:
        # Create track in database first
        song = await create_push_track(spotify_id)

        spotdl_lazy_load()
        spotdl = Spotdl( 
            no_cache=True,
            spotify_client=get_spotipy(),
            downloader_settings=DownloaderOptions(
                format="wav", 
                simple_tui=False,
                print_download_errors=True,
                output=DOWNLOAD_LOC,
                overwrite="skip"
            ),
            loop=asyncio.get_event_loop()
        )
        
        # Fix: search returns a list, get the first item
        songs = spotdl.search([f"https://open.spotify.com/track/{spotify_id}"])
        
        if not songs or len(songs) == 0:
            LOGGER.warning(f"No song found for id: {spotify_id}")
            return

        song_obj = songs[0]  # Get the first (and should be only) song
        LOGGER.info(f"Song found for '{spotify_id}: {song_obj.name} by {song_obj.artist}")
        
        try:
            # Download the song - this also returns a tuple
            result = spotdl.download(song_obj)
            song_result, file_path = result
            
            if not file_path or not os.path.exists(file_path):
                raise DownloaderError(f"Download failed or file not found: {file_path}")
                
        except (LookupError, DownloaderError, AudioProviderError) as e:
            LOGGER.warning(f"{spotify_id} download failed: {str(e)}")

            # Remove from queue on failure
            async with get_session() as s:
                result = await s.execute(delete(song_queue.q_type)
                                         .where(song_queue.q_type.spotify_id == spotify_id))
                await s.commit()

            CURRENTLY_DOWNLOADING.discard(spotify_id)
            return

        # Rename file to include spotify ID for easier management
        old_path = Path(file_path)
        new_name = f"{spotify_id}_{old_path.name}"
        new_path = old_path.parent / new_name
        
        if old_path != new_path and not new_path.exists():
            old_path.rename(new_path)
            file_path = str(new_path)

        file_size = os.path.getsize(file_path)
        LOGGER.info(f"Download completed: {file_path} ({file_size / (1024*1024):.2f} MB)")

        # Add to processing queue
        song_queue.put((file_path, spotify_id))
        CURRENTLY_DOWNLOADING.discard(spotify_id)

        LOGGER.info(f"Successfully downloaded and queued: {song_obj.name}")
        
    except KeyboardInterrupt:
        CURRENTLY_DOWNLOADING.discard(spotify_id)
        raise KeyboardInterrupt
    except Exception as e:
        LOGGER.error(f"Unexpected error downloading {spotify_id}: {traceback.format_exc()}")
        CURRENTLY_DOWNLOADING.discard(spotify_id)

async def start_download_loop(song_queues: list[SongQueue]):
    global DOWNLOAD_LOC
    DOWNLOAD_LOC = "./mock_downloads" if os.getenv("TEST_MODE") else "./downloads"

    LOGGER.info(f"Download loop started ({DOWNLOAD_LOC}).")

    while True:
        await asyncio.sleep(1)
        for q in song_queues:
            try:
                LOGGER.debug(f"{q.name} queue size: {len(q)}")

                if len(q) >= QUEUE_MAX_LEN:
                    continue

                LOGGER.debug(f"{q.name} queue has capacity, ready for new downloads")

                async with get_session() as s:
                    n = QUEUE_MAX_LEN - len(q)

                    queue_items = await s.execute(select(q.q_type.spotify_id)
                                                  .order_by(q.q_type.created_at.asc())
                                                  .limit(n))
                    queue_items = queue_items.scalars().all()
                    LOGGER.debug(f"Found {len(queue_items)} queue items for {q.name}.")

                if len(queue_items) < n:
                    LOGGER.info(F"Queue of {q.name} almost empty, collecting new music (once implemented).")
                    # asyncio.create_task(simple_queue_new_music())

                for db_q_item in list(queue_items):
                    if db_q_item in CURRENTLY_DOWNLOADING:
                        # LOGGER.debug(f"Skipping download for '{q_item}' - already being downloaded.")
                        queue_items.remove(db_q_item)

                    for file_name in os.listdir(DOWNLOAD_LOC):  # Already downloaded.
                        if db_q_item in file_name:
                            if db_q_item not in q:
                                q.put((f"{DOWNLOAD_LOC}/{file_name}", db_q_item))
                                LOGGER.debug(f"Song '{file_name}' was downloaded before, " \
                                             "inserting into local queue.")
                            else:
                                LOGGER.debug(f"Skipping download for '{db_q_item}' - " \
                                             "already downloaded.")

                            queue_items.remove(db_q_item)

                LOGGER.debug(f"Found {len(queue_items)} new items in DB queue for {q.name}: {"', '".join([q_item for q_item in queue_items])}")

                _ = asyncio.gather(*[_download(q_item, q) for q_item in queue_items])
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                LOGGER.error(f"Error in download loop: {traceback.format_exc()}")

    
def clean_downloads(song_queues: list):
    global DOWNLOAD_LOC
    DOWNLOAD_LOC = "./mock_downloads" if os.getenv("TEST_MODE") else "./downloads"

    LOGGER.info(f"Cleaning downloads folder ({DOWNLOAD_LOC}), {len(os.listdir(DOWNLOAD_LOC))} downloaded files.")

    # Get all queue items quickly, minimizing lock time
    protected_ids = set(CURRENTLY_DOWNLOADING)
    for q in song_queues:
        with q.lock:  # Explicit, short lock
            protected_ids.update(x[1] for x in q.queue)
    
    cnt = 0
    # Now process files without holding any locks
    for file in os.listdir(DOWNLOAD_LOC):
        for sp_id in protected_ids:
            LOGGER.debug(f"Checking if {sp_id} in {file}")
            if sp_id in file:
                break
        else:
            LOGGER.debug(f"Deleting song '{file}'.")
            os.remove(os.path.join(DOWNLOAD_LOC, file))
            cnt += 1

    LOGGER.info(f"Finished cleaning downloads folder, " \
                f"removed {cnt} files ({len(os.listdir(DOWNLOAD_LOC))} left).")


async def test():
    # await _download("4sZgFgFZPIwcqDJ4FKJXD2", SongQueue("test", None))

    from models import QueueJukeMIR, QueueAuditus
    async with get_session() as s:
        x = await s.execute(select(QueueAuditus))
        print("X:", x.scalars().all())

if __name__ == "__main__":
    asyncio.run(test())
