"""
Strategies for download_tracking.py's pure planning functions
(plan_downloads/plan_cleanup/match_on_disk/targets_for/resolve_downloads).
The interesting cases are all about *overlap* between wanted_by_queue,
already_local, on_disk, and in_flight - a spotify_id showing up in more than
one of these simultaneously is exactly what plan_downloads/plan_cleanup need
to be fuzzed against, so these strategies should deliberately generate
overlapping sets rather than disjoint random ones.
"""

# spotify_id_pool_strat: a shared small pool of ids (via
# strategies.apis.spotify_id_strat) drawn once per test, then reused across
# wanted_by_queue/already_local/on_disk/in_flight so overlap actually
# happens - independent per-field id generation would rarely collide and
# miss the interesting cases entirely.
# touches: strategies.apis.spotify_id_strat

# wanted_by_queue_strat: dict[queue_name, list[spotify_id]] - queue_name from
# a small fixed set (e.g. "jukemir"/"auditus", matching real SongQueue
# .name values), ids drawn from spotify_id_pool_strat, sometimes overlapping
# across queue names (same id wanted by both).
# touches: collecter.embedders.song_queue.QUEUE_MAX_LEN (as the upper bound
#          on per-queue list size, matching the real caller's contract that
#          wanted_by_queue is already limited to queue capacity)

# already_local_strat: dict[queue_name, set[spotify_id]] - drawn from the
# same pool, sometimes overlapping wanted_by_queue's ids (already-satisfied
# case) and sometimes not.

# on_disk_strat: dict[spotify_id, filepath] - drawn from the same pool,
# filepath shaped like download.py's actual naming convention
# ({spotify_id}_{original_name}).

# download_state_strat: DownloadState with in_flight drawn from the same
# pool, sometimes overlapping wanted_by_queue (already-in-progress case).
# touches: collecter.download_tracking.DownloadState

# protected_and_disk_files_strat: for plan_cleanup specifically - a
# protected_ids set plus a files_on_disk list where some filenames contain a
# protected id (must survive cleanup) and some don't (must be deleted) -
# needs deliberate construction of both cases per draw, not just random
# strings, since plan_cleanup's substring-match logic needs both a hit and a
# miss to be meaningfully exercised.
# touches: collecter.download_tracking.plan_cleanup
