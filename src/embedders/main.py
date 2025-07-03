from logger import LOGGER
import traceback

LOGGER.info("Starting imports...")
from spotdl import Spotdl
from spotdl.types.options import DownloaderOptions

from auditus.transform import AudioArray, AudioLoader, AudioEmbedding, Resampling, Pooling
import librosa as lr
import jukemirlib


from math import floor
import numpy as np
import pandas as pd

import asyncio
import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
from apscheduler.schedulers.asyncio import AsyncIOScheduler

import os


SEGMENT_OVERLAP = 1 
SEGMENT_LENGTH = 24 

SAMPLE_RATE = 16_000  # Jukebox SR=44_100
LAYER = 36


_JUKEMIR_QUEUE = mp.Queue()
def _jukemir_embed(file_path: str) -> np.array:
    LOGGER.debug(f"Starting JukeMIR embedding for file: {file_path}")

    length = floor(lr.get_duration(filename=file_path))
    LOGGER.debug(f"Audio duration detected: {length} seconds for {file_path}")
    
    embeddings = []

    for i, offset in enumerate(range(0, length, SEGMENT_LENGTH - SEGMENT_OVERLAP), start=1):
        segment_duration = min(SEGMENT_LENGTH, length - offset)
        LOGGER.debug(f"Processing segment {i} at offset {offset}s, duration {segment_duration}s")

        audio = jukemirlib.load_audio(file_path,
                                      offset=offset,
                                      duration=segment_duration)
        
        emb = jukemirlib.extract(audio, layers=[LAYER], meanpool=True)[LAYER]
        embeddings.append(emb)

    LOGGER.debug(f"JukeMIR embedding of '{file_path}' successful.")
    return np.array(embeddings)


_AUDITUS_QUEUE = mp.Queue()
def _auditus_embed(file_path: str) -> np.array:
    LOGGER.debug(f"Starting Auditus embedding for file: {file_path}")

    audio = AudioLoader.load_audio(file_path, sr=SAMPLE_RATE)
    # audio = Resampling(target_sr=SAMPLE_RATE)(audio)
    
    embeddings = []
    for i, offset in enumerate(range(0, floor(len(audio.a)/SAMPLE_RATE), 
                                     SEGMENT_LENGTH - SEGMENT_OVERLAP), 
                               start=1):  # Seconds
        
        segment_duration = min(SEGMENT_LENGTH, floor(len(audio.a)/SAMPLE_RATE - offset))
        LOGGER.debug(f"Processing segment {i} at offset {offset}s, duration {segment_duration}s")

        offset_sr = offset * SAMPLE_RATE

        audio_chunk = AudioArray(a=audio.a[offset_sr:offset_sr + SEGMENT_LENGTH*SAMPLE_RATE],
                                 sr=SAMPLE_RATE)

        emb = AudioEmbedding(return_tensors="pt")(audio_chunk)

        emb = Pooling(pooling="mean")(emb)
        embeddings.append(emb.to_numpy())
        
    LOGGER.debug(f"Auditus embedding of '{file_path}' successful.")
    return np.array(embeddings)

def _embed_wrapper(embed: callable, name: str, queue: mp.Queue):
    LOGGER.info(f"{name} embedding loop started.")
    while True:
        try:
            song_file = queue.get(block=True)
            embeddings = embed(song_file)
            LOGGER.info(f"Embedding '{song_file}' using {name} successful.")
            # Push to db
        except:
            LOGGER.info(f"Embedding '{song_file}' using {name} failed: {traceback.format_exc()}")
            # queue.put(song_file, block=True)  # Retry


DOWNLOAD_LOC = "./downloads"
def _sync_download(spotify_url: str) -> str:
    """Run spotdl in a new event loop within the thread"""
    LOGGER.debug(f"Starting synchronous download for: {spotify_url}")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    LOGGER.debug("Initializing Spotdl client")
    spotdl = Spotdl(
        no_cache=True,
        client_id="a766651ba4b744ed82f1e520a75b2455",
        client_secret="767732da0b064b838ebe5d0e3f6ce4eb",
        downloader_settings=DownloaderOptions(format="wav", output=DOWNLOAD_LOC)
    )
    
    LOGGER.debug(f"Searching for song: {spotify_url}")
    song = spotdl.search([spotify_url])
    
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
        LOGGER.warning(f"No song found for URL: {spotify_url}")
        raise Exception("No song found")


async def _download(spotify_url: str):
    LOGGER.info(f"Starting async download process for: {spotify_url}")

    try:
        loop = asyncio.get_event_loop()
            
        LOGGER.debug("Executing download in thread executor")
        with ThreadPoolExecutor() as executor:
            file_path = await loop.run_in_executor(executor, _sync_download, spotify_url)
        
        LOGGER.debug(f"Adding {file_path} to processing queues")
        _JUKEMIR_QUEUE.put(file_path, block=False)
        _AUDITUS_QUEUE.put(file_path, block=False)

        LOGGER.info(f"Downloading song '{file_path}' successful.")
        LOGGER.debug(f"Queue sizes - JukeMIR: {_JUKEMIR_QUEUE.qsize()}, Auditus: {_AUDITUS_QUEUE.qsize()}")
        
    except Exception as e:
        LOGGER.warning(f"Downloading song '{spotify_url}' failed: {traceback.format_exc()}")

async def _download_loop():
    LOGGER.info(f"Download loop started.")

    while True:
        for q, name in [(_JUKEMIR_QUEUE, "JukeMIR"), (_AUDITUS_QUEUE, "Auditus")]:
            LOGGER.debug(f"{name} queue size: {q.qsize()}")
            if q.qsize() > 5:
                continue

            LOGGER.debug(f"{name} queue has capacity, ready for new downloads")

            
            # Get song urls from db
            # LOGGER.info(f"Downloading song '{url}' for {name} queue.")
            # download(url)
        await asyncio.sleep(1)

    
def _clean_downloads():
    LOGGER.info("Cleaning downloads folder.")
    for file in os.listdir(DOWNLOAD_LOC):
        # Note: This file checking logic needs to be fixed - you can't check if file is "in" a queue
        # For now, commenting out the problematic logic
        file_path = os.path.join(DOWNLOAD_LOC, file)
        # if file not in _AUDITUS_QUEUE and file not in _JUKEMIR_QUEUE:
        #     os.remove(file_path)

async def main():
    LOGGER.info("=== Audio Processing Service Starting ===")
    
    # Log system information
    LOGGER.info(f"Python PID: {os.getpid()}")
    LOGGER.info(f"CPU count: {mp.cpu_count()}")
    LOGGER.debug(f"Current working directory: {os.getcwd()}")
    
    try:
        LOGGER.info("Starting embedding worker processes...")

        jukemir_process = mp.Process(
            target=_embed_wrapper, 
            args=(_jukemir_embed, "JukeMIR", _JUKEMIR_QUEUE)
        )
        auditus_process = mp.Process(
            target=_embed_wrapper, 
            args=(_auditus_embed, "Auditus", _AUDITUS_QUEUE)
        )
        
        # jukemir_process.start()
        auditus_process.start()
        
        scheduler = AsyncIOScheduler()
        scheduler.add_job(_clean_downloads, 'interval', hours=1)
        scheduler.start()

        await _download("https://open.spotify.com/track/3dzCClyQ3qKx2o3CLIx02r?si=0c3722a5d8bd4e61")
        
        await _download_loop()
    except Exception as e:
        LOGGER.error(f"Main loop error: {traceback.format_exc()}")
    finally:
        scheduler.shutdown()
        # jukemir_process.terminate()
        auditus_process.terminate()
        # jukemir_process.join()
        auditus_process.join()
    

if __name__ == "__main__":
    asyncio.run(main())
