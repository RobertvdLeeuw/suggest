"""
The one SongQueue guarantee that genuinely can't be tested via
unit/test_song_queue.py's stateful contract tests (real class or double,
both single-process): correctness under actual concurrent OS processes.
Small number of example-based tests, not hypothesis-driven - see our
discussion on why a double can't stand in here (this is specifically about
the multiprocessing plumbing itself, not the logical put/get/remove/peek
contract that's identical whether backed by threading or multiprocessing
primitives).
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
