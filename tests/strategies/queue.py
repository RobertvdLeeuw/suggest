from hypothesis import strategies as st

import asyncio

from tests.strategies.apis import spotify_id_strat

from collecter.embedders import SongQueue, QueueObject
from collecter.downloader import QUEUE_MAX_LEN

from sqlalchemy import delete


@st.composite
def q_item_strat(draw, q_type: QueueObject):
    """Generate Song objects with optional artist relationships."""

    spotify_id = draw(spotify_id_strat())
    song_name = draw(st.text(min_size=1, max_size=100))
    
    return q_type(spotify_id=spotify_id)


@st.composite
def queue_strat(draw, q_type: QueueObject = None, *, 
                min_size=0, max_size=QUEUE_MAX_LEN, fill_via_db=True):
    """Generate queue data only and handle DB stuff in test because async is a no-no in hypothesis."""
    if q_type is None:
        q_type = draw(st.sampled_from(list(QueueObject.__args__)))

    q_items = draw(st.lists(q_item_strat(q_type), min_size=min_size, max_size=max_size, unique=True))
    
    # Return a dict with queue configuration instead of actual queue
    return {
        'q_type': q_type,
        'q_items': q_items,
        'fill_via_db': fill_via_db
    }

async def setup_queue(queue_data, session):
    from collecter.embedders import start_processes
    from collecter.downloader import start_download_loop, clean_downloads, _download
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    
    q = start_processes([queue_data['q_type']])[0]
    
    if queue_data['fill_via_db']:
        await session.execute(delete(q.q_type))
        session.add_all(queue_data['q_items'])
        await session.commit()
        
        asyncio.create_task(start_download_loop([q]))
        
        scheduler = AsyncIOScheduler()
        scheduler.add_job(clean_downloads, 'interval', seconds=0.1, args=([q],))
        scheduler.start()
    else:
        for i, item in enumerate(queue_data['q_items'][:QUEUE_MAX_LEN]):
            await _download(item.spotify_id, q)
    
    return q
