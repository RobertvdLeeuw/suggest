from spotdl import Spotdl

from auditus.transform import AudioArray, AudioLoader, AudioEmbedding, Pooling
import librosa as lr
import jukemirlib


from math import floor
import numpy as np
import pandas as pd

import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import threading
from apscheduler.schedulers.asyncio import AsyncIOScheduler

import os


SEGMENT_OVERLAP = 1 
SEGMENT_LENGTH = 24 

SAMPLE_RATE = 44_100  # Jukebox SR
LAYER = 36

_JUKEMIR_QUEUE = mp.Queue()
def _jukemir_embed(file_path: str) -> np.array:
    length = floor(lr.get_duration(filename=file_path))
    
    embeddings = []

    for offset in range(0, length, SEGMENT_LENGTH - SEGMENT_OVERLAP):
        audio = jukemirlib.load_audio(file_path,  # TODO: This throws errors sometimes, figure out.
                                      offset=offset,
                                      duration=min(SEGMENT_LENGTH, length - offset))
        emb = jukemirlib.extract(audio, layers=[LAYER], meanpool=True)[LAYER]

        embeddings.append(emb)
    return np.array(embeddings)


_AUDITUS_QUEUE = mp.Queue()
def _auditus_embed(file_path: str) -> np.array:
    audio = AudioLoader(sr=SAMPLE_RATE)(file_path)
    
    embeddings = []
    for offset in range(0, len(audio)/SAMPLE_RATE, SEGMENT_LENGTH - SEGMENT_OVERLAP):  # Seconds
        offset_sr = offset * SAMPLE_RATE
        audio_chunk = AudioArray(a=audio.a[offset_sr:offset_sr + SEGMENT_LENGTH*SAMPLE_RATE],
                                 sr=SAMPLE_RATE)

        emb = AudioEmbedding(return_tensors="pt")(audio_chunk)
        emb = Pooling(pooling="mean")(emb)  # JukeMIR meanpools as well, try max at some point?

        embeddings.append(emb.to_numpy())
        
    return np.array(embeddings)

def _embed_wrapper(embed: callable, queue: list):
    while True:
        song_file = queue[0]
        embeddings = embed(song_file)
        # Push to db

        # Remove from db queue
        queue.pop()  # Only after everything passed.

DOWNLOAD_LOC = "./downloads"
_SPOTDL = Spotdl(no_cache=True,
                 output_format="wav",
                 output_directory=DOWNLOAD_LOC)

async def download(spotify_url: str):
    song = _SPOTDL.search(["https://open.spotify.com/track/your_track_id"])
    
    # Figure out object, then some type of no-match check.

    try:
        _, file_path = _SPOTDL.download(song)
        _JUKEMIR_QUEUE.append(file_path)
        _AUDITUS_QUEUE.append(file_path)
    except:
        pass

async def download_loop():
    while True:
        for q in [_JUKEMIR_QUEUE, _AUDITUS_QUEUE]:
            if q.qsize() > 5:
                continue
            
            # Get song urls from db
            # download(url)
        await asyncio.sleep(1)

    
def _clean_downloads():  # This this like every hour
    for file in os.listdir(DOWNLOAD_LOC):
        if file not in _AUDITUS_QUEUE and file not in _JUKEMIR_QUEUE:
            os.remove(file)

async def main():
    with ProcessPoolExecutor(max_workers=2) as executor:
        jukemir_future = executor.submit(embed_wrapper, jukemir_embed)
        auditus_future = executor.submit(embed_wrapper, auditus_embed)
        
        scheduler = AsyncIOScheduler()
        scheduler.add_job(clean_downloads, 'interval', hours=1)
        scheduler.start()
        
        await download_loop(jukemir_queue, auditus_queue)
    

if __name__ == "__main__":
    asyncio.run(main)
