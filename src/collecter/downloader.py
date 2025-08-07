import traceback
import logging
LOGGER = logging.getLogger(__name__)

import os
from pathlib import Path

import asyncio
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select, delete
from db import get_session, setup

from spotdl.types.options import DownloaderOptions

from collecter.embedders import SongQueue, QUEUE_MAX_LEN
from collecter.metadata import (
    simple_queue_new_music, 
    create_push_track,
    _get_sp_album_tracks,
    _add_to_db_queue,
    get_spotipy
)

DOWNLOAD_LOC = None
LookupError = None
Spotdl = None


def spotdl_lazy_load():
    global Spotdl, LookupError

    if Spotdl is not None:
        return

    if os.getenv("TEST_MODE"):
        LOGGER.debug("Initializing mock Spotdl client.")
        from tests.mocks.embedders import Spotdl_fake as _Spotdl, LookupError as _LookupError

    else:
        LOGGER.debug("Initializing Spotdl client.")
        from spotdl import Spotdl as _Spotdl
        from spotdl.download.downloader import LookupError as _LookupError

    Spotdl = _Spotdl
    LookupError = _LookupError

CURRENTLY_DOWNLOADING = set()
async def _download(spotify_id: str, song_queue: SongQueue):#, downloader: Downloader):

    if spotify_id in CURRENTLY_DOWNLOADING:
        return

    for fpath, sp_id in song_queue.peek_all():
        if spotify_id == sp_id:
            LOGGER.debug(f"Song already downloaded: {fpath}.")
            return

    LOGGER.info(f"Starting download for: {spotify_id}")
    CURRENTLY_DOWNLOADING.add(spotify_id)

    try:
        asyncio.create_task(create_push_track(spotify_id))

        spotdl_lazy_load()
        spotdl = Spotdl( 
            no_cache=True,
            spotify_client=get_spotipy(),
            downloader_settings=DownloaderOptions(format="wav", 
                                                  simple_tui=False,
                                                  print_download_errors=False,
                                                  output=DOWNLOAD_LOC),
            loop=asyncio.get_event_loop()
        )
        song = spotdl.search(["https://open.spotify.com/track/" + spotify_id])[0]

        if not song:
            LOGGER.warning(f"No song found for id: {spotify_id}")
            return

        LOGGER.info(f"Song found for '{spotify_id}: {song.name} by {song.artist}")
        
        try:
            _, file_path = spotdl.download(song)
            assert file_path and os.path.exists(file_path)
        except (AssertionError, LookupError):
            LOGGER.info(f"{spotify_id} recognized by Spotdl but audio not found on providers.")

            async with get_session() as s:
                # Remove from queue
                result = await s.execute(delete(song_queue.q_type)
                                         .where(song_queue.q_type.spotify_id == spotify_id))
                await s.commit()

            CURRENTLY_DOWNLOADING.remove(spotify_id)
            return

        old_path = Path(file_path)
        with_id = old_path.parent / f"{spotify_id} {old_path.name}"

        LOGGER.debug(f"Renaming '{old_path}' to '{with_id}'.")
        old_path.rename(with_id)
        file_path = str(with_id)

        file_size = os.path.getsize(file_path)
        LOGGER.debug(f"Download completed: {file_path} ({file_size / (1024*1024):.2f} MB).")

        LOGGER.debug(f"Adding {file_path} to processing queues.")
        song_queue.put((file_path, spotify_id))
        CURRENTLY_DOWNLOADING.remove(spotify_id)

        LOGGER.info(f"Downloading song '{file_path}' successful.")
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        LOGGER.warning(f"Downloading song '{spotify_id}' failed: {traceback.format_exc()}")
        CURRENTLY_DOWNLOADING.remove(spotify_id)

async def start_download_loop(song_queues: list[SongQueue]):
    global DOWNLOAD_LOC
    DOWNLOAD_LOC = "./mock_downloads" if os.getenv("TEST_MODE") else "./downloads"

    LOGGER.info(f"Download loop started.")

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

                LOGGER.debug(f"Found {len(queue_items)} new items in DB queue for {q.name}.")

                _ = asyncio.gather(*[_download(q_item, q) for q_item in queue_items])


            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                LOGGER.error(f"Error in download loop: {traceback.format_exc()}")

    
def clean_downloads(song_queues: list):
    LOGGER.info(f"Cleaning downloads folder ({DOWNLOAD_LOC}), {len(os.listdir(DOWNLOAD_LOC))} downloaded files.")

    cnt = 0
    q_spotify_ids = [x[1] for q in song_queues for x in q.peek_all()]
    for file in os.listdir(DOWNLOAD_LOC):
        for sp_id in set(list(CURRENTLY_DOWNLOADING) + q_spotify_ids):  # Still in the process.
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
    await setup()
    # await _download("4sZgFgFZPIwcqDJ4FKJXD2", SongQueue("test", None))

    from models import QueueJukeMIR, QueueAuditus
    async with get_session() as s:
        x = await s.execute(select(QueueAuditus))
        print("X:", x.scalars().all())

if __name__ == "__main__":
    asyncio.run(test())
