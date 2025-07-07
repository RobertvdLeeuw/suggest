from logger import LOGGER
import traceback

import multiprocessing as mp

from math import floor
import numpy as np

LOGGER.debug(f"Importing JukeMIR Module.")
import librosa as lr
import jukemirlib

LOGGER.debug(f"Importing Auditus Module.")
from auditus.transform import AudioArray, AudioLoader, AudioEmbedding, Resampling, Pooling

SEGMENT_OVERLAP = 1 
SEGMENT_LENGTH = 24 

SAMPLE_RATE = 16_000  # Jukebox SR=44_100
LAYER = 36

MANAGER = mp.Manager()

class SongQueue:  # Queue with peek and membership testing.
    def __init__(self, name: str):
        self.name = name

        manager = mp.Manager()
        self.queue = manager.list()
        self.lock = mp.Lock()
        self.condition = mp.Condition(self.lock)
    
    def put(self, item):
        with self.lock:
            if item not in self.queue:
                self.queue.append(item)
                self.condition.notify()
                return True
            return False
    
    def get(self):
        with self.condition:
            while len(self.queue) == 0:
                self.condition.wait()
            return self.queue.pop(0)
    
    def remove(self, item):
        with self.lock:
            if item in self.queue:
                self.queue.remove(item)
    
    def __contains__(self, item):
        with self.lock:
            return item in self.queue

    def __len__(self):
        return len(self.queue)
    
    def peek(self):
        with self.condition:
            while len(self.queue) == 0:
                self.condition.wait()
            return self.queue[0]


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


def _auditus_embed(file_path: str) -> np.array:
    # Same reason as above.

    LOGGER.debug(f"Starting Auditus embedding for file: {file_path}")

    audio = AudioLoader.load_audio(file_path, sr=SAMPLE_RATE)
    audio = AudioArray(a=np.mean(audio, axis=1), sr=SAMPLE_RATE)  # Stereo -> Mono
    LOGGER.info(audio.a.shape)
    
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
        embeddings.append(emb.numpy())
        
    LOGGER.debug(f"Auditus embedding of '{file_path}' successful.")
    return np.array(embeddings)


def _embed_wrapper(embed_func: callable, name: str, queue: mp.Queue):
    LOGGER.info(f"{name} embedding loop started.")
    while True:
        try:
            song_file = queue.peek()
            embeddings = embed_func(song_file)
            LOGGER.info(f"Embedding '{song_file}' using {name} successful.")
            # Push to db
        except:
            LOGGER.info(f"Embedding '{song_file}' using {name} failed: {traceback.format_exc()}")
        finally:
            queue.get()  # Only remove from queue once fully processed.


EMBEDDERS = [(_jukemir_embed, "JukeMIR"),
             (_auditus_embed, "Auditus")]
PROCESSES = []
def start_processes() -> list[SongQueue]:
    # So we don't wait a shitload when testing smth else which happens to import this script.
    queues = []
    for embed_func, name in EMBEDDERS:
        q = SongQueue(name)
        queues.append(q)

        PROCESSES.append(mp.Process(target=_embed_wrapper, 
                                    args=(embed_func, name, q)))
        PROCESSES[-1].start()
    return queues


def end_processes():
    for p in PROCESSES:
        p.terminate()
        p.join()
