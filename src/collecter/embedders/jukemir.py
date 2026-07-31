"""
Purpose & scope
---------------
JukeMIR embedding - loads audio in SEGMENT_LENGTH-second chunks (with
SEGMENT_OVERLAP seconds of overlap) and extracts a mean-pooled layer-LAYER
representation per chunk via jukemirlib.

jukemirlib (and librosa) are imported lazily in _load(), not at module import
time, since loading them pulls in the full Jukebox checkpoint - a process
that only ever calls auditus.embed() shouldn't pay that cost. jukemirlib
caches its own model as a singleton internally, so unlike auditus.py there's
no separate _MODEL cache needed here.

Does NOT know about Auditus, the DB, or SongQueue - it only turns a file path
into embeddings.
"""

import logging
import os
from math import floor

from models import EmbeddingJukeMIR

LOGGER = logging.getLogger(__name__)

SEGMENT_OVERLAP = 1
SEGMENT_LENGTH = 24
LAYER = 36

_loaded = False
lr = None
jukemirlib = None


def _load() -> None:
    """Lazy load JukeMIR dependencies."""
    global _loaded, lr, jukemirlib
    if _loaded:
        return

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
    _loaded = True
    LOGGER.debug("JukeMIR module loaded successfully")


def embed(file_path: str, song_id: str) -> list[EmbeddingJukeMIR]:
    _load()

    LOGGER.debug(f"Starting JukeMIR embedding for file: {file_path}")

    length = floor(lr.get_duration(filename=file_path))
    LOGGER.debug(f"Audio duration detected: {length} seconds for {file_path}")

    embeddings = []

    for i, offset in enumerate(range(0, length, SEGMENT_LENGTH - SEGMENT_OVERLAP), start=1):
        segment_duration = min(SEGMENT_LENGTH, length - offset)
        LOGGER.debug(
            f"Processing segment {i} at offset {offset}s/{length}s of {file_path}, "
            f"duration {segment_duration}s"
        )
        audio = jukemirlib.load_audio(file_path, offset=offset, duration=segment_duration)

        emb = jukemirlib.extract(audio, layers=[LAYER], meanpool=True)[LAYER]
        embeddings.append(EmbeddingJukeMIR(chunk_id=i, embedding=emb, song_id=song_id))

    LOGGER.info(f"JukeMIR embedding of '{file_path}' successful.")
    return embeddings
