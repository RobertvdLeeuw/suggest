"""
Real multiprocess integration test - moved+renamed from the old
tests/property/test_pipeline.py (that file's only test with actual working
hypothesis code, the rest were commented-out ideas now redistributed across
unit/test_download_tracking.py, unit/test_song_queue.py, and this file).

Needs a real DB (conftest.db_session) AND real worker processes
(embedders.start_processes) - not something unit/ should touch.
"""

# Given a queue with n items + a real embedder process, all items are eventually
# processed (queue DB table drains to 0, in-memory SongQueue empties) AND downloaded
# files get cleaned up afterward.
# touches: collecter.embedders.start_processes, collecter.embedders.end_process,
#          collecter.download.download_loop, collecter.download._clean_downloads,
#          conftest.db_session, conftest.tmp_download_dir, strategies.queue.song_queue_item_strat
# @pytest.mark.slow

# Items removed from an embedder's queue are either embedded WITH song+artist rows
# present in the DB, or failed gracefully (never silently dropped with no trace).
# touches: collecter.embedders (the _async_embed_wrapper loop), conftest.db_session
# @pytest.mark.slow

# Embedding worker processes never deadlock when accessing shared SongQueues under
# real concurrent load from multiple embedders (JukeMIR + Auditus simultaneously).
# touches: collecter.embedders.start_processes, collecter.embedders.song_queue.SongQueue
# @pytest.mark.slow
# NOTE: this is really integration/test_song_queue_concurrency.py's territory (SongQueue
# under real multiprocess load) rather than a separate embedding-pipeline concern -
# consider merging into that file once both are written, to avoid two places asserting
# roughly the same thing with different setup.

# System resource usage (disk space) stays within bounds during a sustained processing
# run - downloaded files never accumulate unbounded even under continuous queueing.
# touches: collecter.download.download_loop, collecter.download._clean_downloads,
#          conftest.tmp_download_dir
# @pytest.mark.slow
