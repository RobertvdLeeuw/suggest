from logger import LOGGER
import traceback
import os

import asyncio
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select, delete
from db import get_session, setup

import nest_asyncio
nest_asyncio.apply()

from spotdl.types.options import DownloaderOptions
if os.getenv("TEST_MODE"):
    # from mocks import Spotdl
else:
    from spotdl import Spotdl

from embedders import SongQueue
from metadata import simple_queue_new_music


DOWNLOAD_LOC = "./downloads"
LOGGER.debug("Initializing Spotdl client")

from metadata import (simple_queue_new_music, 
    create_push_track,
    _get_sp_album_tracks,
    _add_to_db_queue,
    get_spotipy
)


async def _download(spotify_id: str, song_queue: SongQueue):#, downloader: Downloader):
    LOGGER.info(f"Starting download for: {spotify_id}")

    try:
        await create_push_track(spotify_id)  # TODO: Background task

        for fpath, sp_id in song_queue.peek_all():
            if spotify_id == sp_id:
                LOGGER.debug(f"Song already downloaded: {fpath}.")
                return

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
        
        _, file_path = spotdl.download(song)

        if not file_path or not os.path.exists(file_path):
            LOGGER.info(f"{spotify_id} recognized by Spotdl but audio not found on providers.")
            async with get_session() as s:
                # Remove from queue
                result = await s.execute(delete(song_queue.q_type)
                                         .where(song_queue.q_type.spotify_id == spotify_id))
                await s.commit()
            return

        file_size = os.path.getsize(file_path)
        LOGGER.debug(f"Download completed: {file_path} ({file_size / (1024*1024):.2f} MB).")

        LOGGER.debug(f"Adding {file_path} to processing queues.")
        song_queue.put((file_path, spotify_id))

        LOGGER.info(f"Downloading song '{file_path}' successful.")
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        LOGGER.warning(f"Downloading song '{spotify_id}' failed: {traceback.format_exc()}")
    # return True

QUEUE_MAX_LEN = 5
async def start_download_loop(song_queues: list[SongQueue]):
    LOGGER.info(f"Download loop started.")

    while True:
        for q in song_queues:
            await asyncio.sleep(30)

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
                LOGGER.debug(f"Found {len(queue_items)} items in DB queue for {q.name}.")

                _ = asyncio.gather(*[_download(q_item, q) for q_item in queue_items])

                if len(queue_items) < n:
                    LOGGER.info(F"Queue of {q.name} almost empty, collecting new music (once implemented).")
                    # await _add_to_db_queue(_get_sp_album_tracks("2zQeigA4bFAlTqQqBiVe6Y"))
                    # await simple_queue_new_music()  # TODO: Background task

            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                LOGGER.error(f"Error in download loop: {traceback.format_exc()}")

    
def clean_downloads(song_queues: list):
    LOGGER.info("Cleaning downloads folder.")
    for file in os.listdir(DOWNLOAD_LOC):
        for q in song_queues:
            if file in q:
                break
        else:
            LOGGER.debug(f"Deleting song '{file}'.")
            os.remove(os.path.join(DOWNLOAD_LOC, file))
    LOGGER.info("Finished cleaning downloads folder.")


async def test():
    await setup()
    # await _download("4sZgFgFZPIwcqDJ4FKJXD2", SongQueue("test", None))

    from models import QueueJukeMIR, QueueAuditus
    async with get_session() as s:
        x = await s.execute(select(QueueAuditus))
        print("X:", x.scalars().all())

if __name__ == "__main__":
    asyncio.run(test())
