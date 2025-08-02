from hypothesis import given

from tests.strategies.queue import queue
from tests.strategies.apis import spotify_id_strat, q_item_strat
from tests.mocks.db import clean_session

from collecter.embedders import SongQueue, start_processes, QueueObject
from sqlalchemy import select, delete

import time
import asyncio
import os
os.environ["TEST_MODE"] = "true"
os.environ["POSTGRES_DB"] = "test_collecter"


async def setup_queue(q_items: list, session) -> tuple[SongQueue, QueueObject]:
    q_type = type(q_items[0])
    q = await start_processes([q_type])

    clean_session.add_all(q_items)
    clean_session.commit()

    asyncio.create_task(start_download_loop([q]))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(clean_downloads, 'interval', seconds=1, args=([q],))
    scheduler.start()

    return q, q_type

async def n_items_in_queue_table(q_type: QueueObject):
    return await clean_session.execute(select(q_type.spotify_id).count())

from downloader import QUEUE_MAX_LEN

@given(st.list(q_item_strat(), max_size=QUEUE_MAX_LEN+10, unique=True))
async def queue_processes_all_and_cleans(q_items):
    """Given a queue with n items + embedder, all items will be processed eventually 
       AND downloaded files will be removed."""

    q, q_type = await setup_queue(q_items, clean_session)
    asyncio.create_task(start_download_loop([q]))

    timeout = time.time() + 30
    while time.time() < timeout:
        if n_items_in_queue_table(q_type) == 0: break
        await asyncio.sleep(0.1)

    assert n_items_in_queue_table(q_type) == 0
    assert len(q) == 0
    
    await asyncio.sleep(3)
    assert len(os.listdir("./mock_downloads")) == 0

@given(st.list(q_item_strat(), max_size=10, unique=True))
async def queue_items_saved_or_caught(q_items):
    """Items removed from queue are either embedded WITH song+artist objects or failed gracefully."""

    pass

@given(st.list(q_item_strat(), max_size=QUEUE_MAX_LEN+10, unique=True))
async def queue_never_over_max(q_items, clean_session): # TODO: Do I have to pass clean_session to in-prod, or is that handled for us?
    """Queue never exceeds max len."""
    q, q_type = await setup_queue(q_items, clean_session)
    asyncio.create_task(start_download_loop([q]))

    while await n_items_in_queue_table(q_type) > 0:
        assert len(q) >= QUEUE_MAX_LEN

# QueueItem types always align with expected type AND expected embedder.

# Files successfully downloaded always have valid audio format and non-zero size

# Embedding processes never deadlock when accessing shared queues

# Downloaded files are never deleted while still being processed by any embedder

# System resource usage (disk space, memory) stays within bounds during processing

