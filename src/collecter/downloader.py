from logger import LOGGER
import traceback

from spotdl import Spotdl
from spotdl.types.options import DownloaderOptions

from embedders import SongQueue
from metadata import simple_queue_new_music


DOWNLOAD_LOC = "./downloads"
def _sync_download(spotify_id: str) -> str:
    """Run spotdl in a new event loop within the thread"""
    LOGGER.debug(f"Starting synchronous download for: {spotify_id}")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    LOGGER.debug("Initializing Spotdl client")
    spotdl = Spotdl(
        no_cache=True,
        client_id="a766651ba4b744ed82f1e520a75b2455",
        client_secret="767732da0b064b838ebe5d0e3f6ce4eb",
        downloader_settings=DownloaderOptions(format="wav", output=DOWNLOAD_LOC)
    )
    
    LOGGER.debug(f"Searching for song: {spotify_id}")
    song = spotdl.search([spotify_id])
    
    if song:
        LOGGER.info(f"Song found: {song[0].name} by {song[0].artist}")
        
        _, file_path = spotdl.download(song[0])
        
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            LOGGER.info(f"Download completed: {file_path} ({file_size / (1024*1024):.2f} MB).")
        else:
            LOGGER.error(f"Download completed but file not found: {file_path}")
            
        return file_path
    else:
        LOGGER.warning(f"No song found for id: {spotify_id}")
        raise Exception("No song found")


async def _download(spotify_id: str):
    LOGGER.info(f"Starting async download process for: {spotify_id}")

    try:
        loop = asyncio.get_event_loop()
            
        LOGGER.debug("Executing download in thread executor")
        with ThreadPoolExecutor() as executor:
            file_path = await loop.run_in_executor(executor, _sync_download, spotify_id)
        
        LOGGER.debug(f"Adding {file_path} to processing queues")
        _JUKEMIR_QUEUE.put(file_path)
        _AUDITUS_QUEUE.put(file_path)

        LOGGER.info(f"Downloading song '{file_path}' successful.")
        
    except Exception as e:
        LOGGER.warning(f"Downloading song '{spotify_id}' failed: {traceback.format_exc()}")

async def start_download_loop(song_queues: list):
    LOGGER.info(f"Download loop started.")

    while True:
        for q in song_queues:
            LOGGER.debug(f"{q.name} queue size: {len(q)}")
            if len(q) > 5:
                continue

            LOGGER.debug(f"{q.name} queue has capacity, ready for new downloads")
            
            # Get song urls from db
                # If queue empty, simple_queue_new_music()
            # LOGGER.info(f"Downloading song '{url}' for {name} queue.")
            # Collect artist and song metadata here in non-blocking call.
            # download(url)
            # Append to queue
        await asyncio.sleep(30)

    
def clean_downloads(song_queues: list):
    LOGGER.info("Cleaning downloads folder.")
    for file in os.listdir(DOWNLOAD_LOC):
        for q in song_queues:
            if file in q:
                break
        else:
            LOGGER.debug(f"Deleting song '{file}'.")
            os.remove(os.path.join(DOWNLOAD_LOC, file))
    LOGGER.debug("Finished cleaning downloads folder.")

