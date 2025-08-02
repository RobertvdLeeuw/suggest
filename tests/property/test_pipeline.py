import pytest
import asyncio
import time
import os
from hypothesis import given, settings
from sqlalchemy import select

from tests.strategies.queue import queue_strat, setup_queue
from tests.strategies.apis import spotify_id_strat
from collecter.embedders import SongQueue, start_processes, QueueObject
from collecter.downloader import QUEUE_MAX_LEN
from models import QueueAuditus, QueueJukeMIR

os.environ["TEST_MODE"] = "true"
os.environ["POSTGRES_DB"] = "test_collecter"

async def n_items_in_queue_table(q_type: QueueObject, session):
    result = await session.execute(select(q_type.spotify_id).count())
    return result.scalar()

pytestmark = pytest.mark.collecter

@pytest.mark.asyncio
@pytest.mark.slow
@given(q_data=queue_strat(max_size=QUEUE_MAX_LEN+10))
@settings(max_examples=10, deadline=60000)  # Increase deadline for async operations
async def test_queue_processes_all_and_cleans(q_data, clean_session):
    """Given a queue with n items + embedder, all items will be processed eventually 
       AND downloaded files will be removed."""

    q = await setup_queue(q_data, clean_session)

    timeout = time.time() + 30
    while time.time() < timeout:
        count = await n_items_in_queue_table(q.q_type, clean_session)
        if count == 0:
            break
        await asyncio.sleep(0.1)

    final_count = await n_items_in_queue_table(q.q_type, clean_session)
    assert final_count == 0
    assert len(q) == 0
    
    await asyncio.sleep(3)
    assert len(os.listdir("./mock_downloads")) == 0

# @given(queue_strat(fill_via_db=False))
# @settings(max_examples=5)
# @pytest.mark.asyncio
# async def test_queue_items_saved_or_caught(q_data, clean_session):
#     """Items removed from queue are either embedded WITH song+artist objects or failed gracefully."""
#     pass

# @given(queue_strat(max_size=QUEUE_MAX_LEN+10))
# @settings(max_examples=10, deadline=60000)
# @pytest.mark.asyncio
# @pytest.mark.slow
# async def test_queue_never_over_max(q_data, clean_session): 
#     """Queue never exceeds max len."""

#     q = await setup_queue(q_data, clean_session)
    
#     while True:
#         count = await n_items_in_queue_table(q.q_type, clean_session)
#         if count == 0:
#             break
#         assert len(q) <= QUEUE_MAX_LEN
#         await asyncio.sleep(0.1)

# @pytest.mark.asyncio
# async def test_embedding_processes_no_deadlock():
#     """Embedding processes never deadlock when accessing shared queues"""
#     pass

# @pytest.mark.asyncio
# async def test_file_deletion_safety():
#     """Downloaded files are never deleted while still being processed by any embedder"""
#     pass

# @pytest.mark.asyncio
# async def test_resource_usage_bounds():
#     """System resource usage (disk space, memory) stays within bounds during processing"""
#     pass
