"""
Strategies for SongQueue operation sequences, used by unit/test_song_queue.py's
stateful property tests (see that file's module docstring) and by
integration/test_song_queue_concurrency.py's real-process stress test.

The pre-refactor version of this file imported collecter.downloader's
start_download_loop/clean_downloads/_download and collecter.embedders'
start_processes directly - none of those names exist anymore post-refactor
(split across download.py/download_tracking.py/embedders/__init__.py now).
Rebuilt clean rather than patched.
"""

# song_queue_item_strat: a single (filepath, spotify_id) tuple, matching what
# download.py actually .put()s onto a SongQueue (see _download_one/plan_downloads'
# queue_puts) - NOT a bare spotify_id or a Song/QueueObject ORM row, that was
# the old strategy's shape and doesn't match current SongQueue usage.
# touches: strategies.apis.spotify_id_strat

# operation_sequence_strat: a list of operations (each one of
# put(item)/get()/remove(item)/peek()/peek_all()) for
# unit/test_song_queue.py's RuleBasedStateMachine to replay - only needed if
# the state machine doesn't generate its own rule sequence via hypothesis's
# built-in stateful machinery (RuleBasedStateMachine normally handles
# sequencing itself via @rule decorators, so this may not be needed at all -
# confirm once test_song_queue.py's actual structure is written before
# building this).

<<<<<<< HEAD
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
    from apscheduler.schedulers.background import BackgroundScheduler
    
    q = start_processes([queue_data['q_type']])[0]
    
    if queue_data['fill_via_db']:
        await session.execute(delete(q.q_type))
        session.add_all(queue_data['q_items'])
        await session.commit()
    else:
        for i, item in enumerate(queue_data['q_items'][:QUEUE_MAX_LEN]):
            await _download(item.spotify_id, q)
    
    clean_downloads([q])
    asyncio.create_task(start_download_loop([q]))

    scheduler = BackgroundScheduler()
    job = scheduler.add_job(clean_downloads, 'interval', seconds=1, args=([q],))
    scheduler.start()

    q._scheduler = scheduler
    q._job = job

    return q

    # yield q
    # scheduler.shutdown()
=======
# queue_name_strat: small fixed set matching real SongQueue .name values used
# elsewhere (e.g. "jukemir"/"auditus") - kept here rather than duplicated in
# strategies/download.py's wanted_by_queue_strat; that one should import this.
>>>>>>> tests
