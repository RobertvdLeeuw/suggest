"""
Purpose & scope
---------------
Auditus embedding - resamples audio to mono/SAMPLE_RATE, then extracts a
mean-pooled embedding per SEGMENT_LENGTH-second chunk (with SEGMENT_OVERLAP
seconds of overlap) via auditus's AudioEmbedding model.

auditus is imported lazily in _load(), not at module import time, since
AudioEmbedding() loads the full model into memory - a process that only ever
calls jukemir.embed() shouldn't pay that cost. _MODEL is cached at module
level (unlike JukeMIR, which caches its model as a singleton inside
jukemirlib itself) because AudioEmbedding() reloads the whole model on every
call otherwise.

Does NOT know about JukeMIR, the DB, or SongQueue - it only turns a file path
into embeddings.
"""

import logging
import os
from math import floor

import numpy as np

from models import EmbeddingAuditus

LOGGER = logging.getLogger(__name__)

SEGMENT_OVERLAP = 1
SEGMENT_LENGTH = 24
SAMPLE_RATE = 16_000  # Target rate audio is resampled to before embedding

_loaded = False
AudioArray = None
AudioLoader = None
AudioEmbedding = None
Resampling = None
Pooling = None
_MODEL = None


def _load() -> None:
    """Lazy load Auditus dependencies."""
    global _loaded, AudioArray, AudioLoader, AudioEmbedding, Resampling, Pooling
    if _loaded:
        return

    if os.getenv("TEST_MODE"):
        LOGGER.debug("Loading mock Auditus module...")
        from tests.mocks.embedders import auditus_fake as _auditus

        AudioEmbedding = _auditus().AudioEmbedding
        AudioLoader = _auditus().AudioLoader()
    else:
        LOGGER.debug("Loading (actual) Auditus module...")
        from auditus.transform import AudioEmbedding as _AudioEmbedding
        from auditus.transform import AudioLoader as _AudioLoader

        AudioEmbedding = _AudioEmbedding
        AudioLoader = _AudioLoader

    from auditus.transform import AudioArray as _AudioArray
    from auditus.transform import Pooling as _Pooling
    from auditus.transform import Resampling as _Resampling

    AudioArray = _AudioArray
    Resampling = _Resampling
    Pooling = _Pooling
    _loaded = True
    LOGGER.debug("Auditus module loaded successfully")


def embed(file_path: str, song_id: str) -> list[EmbeddingAuditus]:
    global _MODEL
    _load()

    LOGGER.debug(f"Starting Auditus embedding for file: {file_path}")

    audio = AudioLoader.load_audio(file_path)
    audio = AudioArray(a=np.mean(audio, axis=1), sr=audio.sr)  # Stereo -> Mono
    audio = Resampling(target_sr=SAMPLE_RATE)(audio)

    length = floor(len(audio.a) / SAMPLE_RATE)

    embeddings = []
    for i, offset in enumerate(
        range(0, length, SEGMENT_LENGTH - SEGMENT_OVERLAP), start=1
    ):  # Seconds
        segment_duration = min(SEGMENT_LENGTH, floor(len(audio.a) / SAMPLE_RATE - offset))
        LOGGER.debug(
            f"Processing segment {i} at offset {offset}s/{length}s of {file_path}, "
            f"duration {segment_duration}s"
        )
        offset_sr = offset * SAMPLE_RATE

        audio_chunk = AudioArray(
            a=audio.a[offset_sr : offset_sr + SEGMENT_LENGTH * SAMPLE_RATE], sr=SAMPLE_RATE
        )

        if _MODEL is None:
            # This loads the entire model every time, so we need to save it (whereas JukeMIR is a singleton).
            _MODEL = AudioEmbedding(return_tensors="pt")

        emb = _MODEL(audio_chunk)

        emb = Pooling(pooling="mean")(emb)
        embeddings.append(EmbeddingAuditus(chunk_id=i, embedding=emb.numpy(), song_id=song_id))

    LOGGER.info(f"Auditus embedding of '{file_path}' successful.")
    return embeddings
