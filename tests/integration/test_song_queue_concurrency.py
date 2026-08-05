"""
The SongQueue guarantees that genuinely can't be tested via
unit/test_song_queue.py's stateful contract test (SongQueueDouble,
single-process): correctness under actual concurrent OS processes. Small
number of example-based tests, not hypothesis-driven - this is specifically
about the multiprocessing plumbing itself (Manager().list() proxying,
Lock/Condition across process boundaries), not the logical
put/get/remove/peek contract that's identical whether backed by threading
or multiprocessing primitives and is already covered by the double's
stateful test.

Also owns the "no deadlock across multiple real processes sharing a
SongQueue" assertion that used to live in test_embedding_pipeline.py -
moved here since it's a SongQueue-under-load question, not an
embedding-pipeline one; exercised with multiple producer/consumer
processes shaped like real embedder workers rather than duplicating the
full embedding pipeline's setup just to get concurrent processes.
"""

# N real processes calling put() concurrently with overlapping items never lose an
# item and never duplicate one - final queue contents match the deduplicated union of
# everything put, regardless of interleaving.
# touches: collecter.embedders.song_queue.SongQueue (real class, real mp.Process workers)
# @pytest.mark.slow

# Concurrent put()/get() across real processes never deadlocks - a bounded-time test
# (explicit timeout, fail rather than hang forever) with multiple producer and
# consumer processes hammering the same SongQueue.
# touches: collecter.embedders.song_queue.SongQueue
# @pytest.mark.slow

# get()'s 30s timeout-then-raise behavior actually fires under real multiprocess
# conditions when no producer ever puts anything - confirms the TimeoutError contract
# holds across process boundaries, not just in-process.
# touches: collecter.embedders.song_queue.SongQueue
# @pytest.mark.slow

# Multiple embedder-shaped processes (JukeMIR + Auditus stand-ins) sharing SongQueues
# never deadlock under real concurrent put/get load - bounded-time test, fail rather
# than hang. (Moved from test_embedding_pipeline.py - see module docstring.)
# touches: collecter.embedders.song_queue.SongQueue, collecter.embedders.start_processes
# @pytest.mark.slow
