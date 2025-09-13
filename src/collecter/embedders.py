import traceback
import logging
LOGGER = logging.getLogger(__name__)

import time
import os

import multiprocessing as mp
from queue import Queue
import asyncio

from sqlalchemy import select, delete, exists

from models import EmbeddingJukeMIR, EmbeddingAuditus, QueueJukeMIR, QueueAuditus, Song
from db import get_session
from collecter.metadata import create_push_track

from math import floor
import numpy as np

SEGMENT_OVERLAP = 1 
SEGMENT_LENGTH = 24 

SAMPLE_RATE = 16_000  # Jukebox SR=44_100
LAYER = 36

# Global variables to cache imports
_jukemir_loaded = False
_auditus_loaded = False
lr = None
jukemirlib = None
AudioArray = None
AudioLoader = None
AudioEmbedding = None
Resampling = None
Pooling = None

def _jukemir_load():
    """Lazy load JukeMIR dependencies"""
    global _jukemir_loaded, lr, jukemirlib
    if not _jukemir_loaded:
        import librosa as _lr

        if os.getenv("TEST_MODE"):
            LOGGER.debug("Loading mock JukeMIR module...")
            from tests.mocks.embedders import jukemirlib_fake as _jukemirlib
            jukemirlib = _jukemirlib()
        else:
            LOGGER.debug("Loading (actual) JukeMIR module...")
            import jukemirlib as _jukemirlib
            jukemirlib = _jukemirlib
        
        lr = _lr
        _jukemir_loaded = True
        LOGGER.debug("JukeMIR Module loaded successfully")

def _auditus_load():
    """Lazy load Auditus dependencies"""
    global _auditus_loaded, AudioArray, AudioLoader, AudioEmbedding, Resampling, Pooling
    if not _auditus_loaded:
        if os.getenv("TEST_MODE"):
            LOGGER.debug("Loading mock Auditus module...")
            from tests.mocks.embedders import auditus_fake as _auditus
            AudioEmbedding = _auditus().AudioEmbedding
            AudioLoader = _auditus().AudioLoader()
        else:
            LOGGER.debug("Loading (actual) Auditus module...")
            from auditus.transform import AudioEmbedding as _AudioEmbedding, AudioLoader as _AudioLoader
            AudioEmbedding = _AudioEmbedding
            AudioLoader = _AudioLoader

        from auditus.transform import AudioArray as _AudioArray, Resampling as _Resampling, Pooling as _Pooling
    
        AudioArray = _AudioArray
        Resampling = _Resampling
        Pooling = _Pooling
        _auditus_loaded = True
        LOGGER.debug("Auditus Module loaded successfully")


QueueObject = QueueJukeMIR | QueueAuditus

QUEUE_MAX_LEN = 5
class SongQueue:  # Queue with peek and membership testing.
    def __init__(self, name: str, q_type: QueueObject):
        self.name = name
        self.q_type = q_type

        self._process: mp.Process

        self.queue = mp.Manager().list()
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
        with self.lock:
            return len(self.queue)

    def peek(self):
        while True:
            with self.condition:
                if len(self.queue) > 0:
                    return self.queue[0]
            time.sleep(1)  # So we don't hog the lock while waiting.

    def peek_all(self):
        with self.lock:
            return list(self.queue)


def _jukemir_embed(file_path: str, song_id: str) -> list[EmbeddingJukeMIR]:
    _jukemir_load()

    LOGGER.debug(f"Starting JukeMIR embedding for file: {file_path}")

    length = floor(lr.get_duration(filename=file_path))
    LOGGER.debug(f"Audio duration detected: {length} seconds for {file_path}")
    
    embeddings = []

    for i, offset in enumerate(range(0, length, SEGMENT_LENGTH - SEGMENT_OVERLAP), start=1):
        segment_duration = min(SEGMENT_LENGTH, length - offset)
        LOGGER.debug(f"Processing segment {i} at offset {offset}s/{length}s of {file_path}, " \
                     f"duration {segment_duration}s")
        audio = jukemirlib.load_audio(file_path,
                                      offset=offset,
                                      duration=segment_duration)
        
        emb = jukemirlib.extract(audio, layers=[LAYER], meanpool=True)[LAYER]
        embeddings.append(EmbeddingJukeMIR(chunk_id=i, embedding=emb, song_id=song_id))

    LOGGER.info(f"JukeMIR embedding of '{file_path}' successful.")
    return embeddings

_AUDITUS_MODEL = None
def _auditus_embed(file_path: str, song_id: str) -> list[EmbeddingAuditus]:
    global _AUDITUS_MODEL
    _auditus_load()

    LOGGER.debug(f"Starting Auditus embedding for file: {file_path}")

    # audio = AudioLoader.load_audio(file_path, sr=SAMPLE_RATE)
    audio = AudioLoader.load_audio(file_path)
    audio = AudioArray(a=np.mean(audio, axis=1), sr=audio.sr)  # Stereo -> Mono
    audio = Resampling(target_sr=SAMPLE_RATE)(audio)

    length = floor(len(audio.a)/SAMPLE_RATE)

    embeddings = []
    for i, offset in enumerate(range(0, length, 
                                     SEGMENT_LENGTH - SEGMENT_OVERLAP), 
                               start=1):  # Seconds
        
        segment_duration = min(SEGMENT_LENGTH, floor(len(audio.a)/SAMPLE_RATE - offset))
        LOGGER.debug(f"Processing segment {i} at offset {offset}s/{length}s of {file_path}, " \
                     f"duration {segment_duration}s")
        offset_sr = offset * SAMPLE_RATE


        audio_chunk = AudioArray(a=audio.a[offset_sr:offset_sr + SEGMENT_LENGTH*SAMPLE_RATE],
                                 sr=SAMPLE_RATE)

        if _AUDITUS_MODEL is None:
            # This loads the entire model every time, so we need to save it (whereas JukeMIR is a singleton).
            _AUDITUS_MODEL = AudioEmbedding(return_tensors="pt") 

        emb = _AUDITUS_MODEL(audio_chunk)

        emb = Pooling(pooling="mean")(emb)
        embeddings.append(EmbeddingAuditus(chunk_id=i, embedding=emb.numpy(), song_id=song_id))
        
    LOGGER.info(f"Auditus embedding of '{file_path}' successful.")
    return embeddings

async def _async_embed_wrapper(embed_func: callable, name: str, queue: Queue, emb_type):
    LOGGER.info(f"{name} embedding loop started.")
    while True:
        song_file = None
        spotify_id = None

        try:
            song_file, spotify_id = queue.peek()
            
            async with get_session() as s:
                song = await create_push_track(spotify_id)

                result = await s.execute(select(exists().where(emb_type.song_id == song.song_id)))
                if not result.scalar():
                    LOGGER.debug(f"Start embed, {name}, {song.song_name}.")
                    embeddings = embed_func(song_file, song.song_id)
                    s.add_all(embeddings)
                else:
                    LOGGER.warning(f"About to embed {song.song_name} using {name}, " \
                                   "but it's already embedded.")

                # Remove from queue
                result = await s.execute(delete(queue.q_type)
                                         .where(queue.q_type.spotify_id == spotify_id))
                
                await s.commit()
                LOGGER.debug(f"Pushed {name} embeddings of '{song_file}' to DB.")

            LOGGER.info(f"Embedding '{song_file}' using {name} successful.")
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            LOGGER.warning(f"Embedding '{song_file}' using {name} failed: {traceback.format_exc()}")

        queue.get()

def _embed_wrapper(embed_func: callable, name: str, queue: Queue, emb_type):
    """Synchronous wrapper that runs the async embed_wrapper in an event loop"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        loop.run_until_complete(_async_embed_wrapper(embed_func, name, queue, emb_type))
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        LOGGER.error(f"Process {name} failed: {traceback.format_exc()}")
    finally:
        if 'loop' in locals():
            loop.close()

EMBEDDERS = [(_jukemir_embed, "JukeMIR", QueueJukeMIR, EmbeddingJukeMIR),
             (_auditus_embed, "Auditus", QueueAuditus, EmbeddingAuditus)]
PROCESSES = []
def start_processes(selection: list[QueueObject] = []) -> list[SongQueue]:
    queues = []
    for embed_func, name, q_type, emb_type in EMBEDDERS:
        if selection and q_type not in selection:
            continue

        q = SongQueue(name, q_type)
        queues.append(q)

        process = mp.Process(
            target=_embed_wrapper, 
            args=(embed_func, name, q, emb_type)
        )
        PROCESSES.append(process)
        process.start()
        q._process = process

    return queues

def end_processes():
    for p in PROCESSES:
        end_process(p)
    PROCESSES.clear()

def end_process(p):
    if p.is_alive():
        p.terminate()
        p.join(timeout=5) 

        if p.is_alive():
            LOGGER.warning(f"Process {p.name} did not terminate gracefully, forcing kill")
            p.kill()
