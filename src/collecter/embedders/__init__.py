"""
Purpose & scope
---------------
Process orchestration for the embedding pipeline: pairs each embedder
(jukemir.embed, auditus.embed) with its queue/embedding types in EMBEDDERS,
and runs each one in its own subprocess pulling from a SongQueue.

jukemir.py and auditus.py never import each other or this module - this is
the only file that knows both exist, so adding a third embedder means adding
one entry to EMBEDDERS here, not touching the other two.

Re-exports QUEUE_MAX_LEN, QueueObject, and SongQueue from .song_queue so
`from collecter.embedders import SongQueue` etc. keeps working for callers
that don't need to care this is a package now.
"""

import asyncio
import logging
import multiprocessing as mp
import traceback

from ...db import get_session
from ...models import EmbeddingAuditus, EmbeddingJukeMIR, QueueAuditus, QueueJukeMIR
from ..repository import SqlAlchemyRepository
from . import auditus, jukemir
from .song_queue import QUEUE_MAX_LEN, QueueObject, SongQueue

LOGGER = logging.getLogger(__name__)

__all__ = [
    "QUEUE_MAX_LEN",
    "QueueObject",
    "SongQueue",
    "start_processes",
    "end_processes",
    "end_process",
]


async def _async_embed_wrapper(embed_func: callable, name: str, queue: SongQueue, emb_type):
    LOGGER.info(f"{name} embedding loop started.")
    while True:
        song_file = None
        spotify_id = None

        try:
            song_file, spotify_id = queue.peek()

            async with get_session() as s:
                repo = SqlAlchemyRepository(s)

                song = await repo.get_song_by_spotify_id(spotify_id)
                if song is None:
                    # Shouldn't happen: the downloader pushes the Song row before
                    # ever queueing it (see old/downloader.py's ordering) - a miss
                    # here means that invariant was violated somewhere upstream.
                    # No song_id to embed against, so there's nothing to do but
                    # drop it from the queue and move on.
                    LOGGER.warning(
                        f"{name}: no Song row for spotify_id={spotify_id}, "
                        f"expected it to already be pushed. Dropping from queue."
                    )
                    await repo.dequeue_track(queue.q_type, spotify_id)
                    queue.get()
                    continue

                if not await repo.is_song_embedded(song.song_id, emb_type):
                    LOGGER.debug(f"Start embed, {name}, {song.song_name}.")
                    embeddings = embed_func(song_file, song.song_id)
                    await repo.save_embeddings(embeddings)
                else:
                    LOGGER.warning(
                        f"About to embed {song.song_name} using {name}, but it's already embedded."
                    )

                await repo.dequeue_track(queue.q_type, spotify_id)
                LOGGER.debug(f"Pushed {name} embeddings of '{song_file}' to DB.")

            LOGGER.info(f"Embedding '{song_file}' using {name} successful.")
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            LOGGER.warning(f"Embedding '{song_file}' using {name} failed: {traceback.format_exc()}")

        queue.get()


def _embed_wrapper(embed_func: callable, name: str, queue: SongQueue, emb_type):
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
        if "loop" in locals():
            loop.close()


EMBEDDERS = [
    (jukemir.embed, "JukeMIR", QueueJukeMIR, EmbeddingJukeMIR),
    (auditus.embed, "Auditus", QueueAuditus, EmbeddingAuditus),
]
PROCESSES = []


def start_processes(selection: list[QueueObject] = []) -> list[SongQueue]:
    queues = []
    for embed_func, name, q_type, emb_type in EMBEDDERS:
        if selection and q_type not in selection:
            continue

        q = SongQueue(name, q_type)
        queues.append(q)

        process = mp.Process(target=_embed_wrapper, args=(embed_func, name, q, emb_type))
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
