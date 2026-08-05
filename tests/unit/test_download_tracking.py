"""
download_tracking.py's planning functions (plan_downloads/plan_cleanup/
match_on_disk/targets_for/resolve_downloads), tested with hand-picked
example cases rather than a dedicated Hypothesis strategy module.

Scope note (post-slimdown): this file previously depended on
strategies/download.py for generated wanted_by_queue/already_local/on_disk/
in_flight overlap scenarios. The overlap cases these functions care about
are few and enumerable (an id wanted-and-on-disk, wanted-and-in-flight,
wanted-by-two-queues, a filename that's a substring collision, etc) - a
handful of module-level constants covering each case does the same job as
a generated strategy here without the strategy-maintenance overhead. Kept
Hypothesis strategies for resolution.py/retry.py/listen_tracking.py where
input shape is genuinely wide; this module's inputs are small enough that
"did we cover the actual overlap cases" is easy to check by eye against a
fixed set.
"""

# Module-level example fixtures (small, fixed, deliberately overlapping):
# WANTED_BY_QUEUE = {"jukemir": {id_a, id_b}, "auditus": {id_b, id_c}}
# ALREADY_LOCAL = {"jukemir": {id_a}}
# ON_DISK = {id_a: f"{id_a}_track.mp3"}
# IN_FLIGHT = {id_c}
# touches: collecter.download_tracking.DownloadState

# plan_downloads never proposes a spotify_id that match_on_disk already found on disk -
# it goes into queue_puts instead of ids_to_download.
# touches: collecter.download_tracking.plan_downloads, collecter.download_tracking.match_on_disk

# plan_downloads never re-proposes a spotify_id already in state.in_flight from a
# previous tick.
# touches: collecter.download_tracking.plan_downloads, collecter.download_tracking.DownloadState

# plan_downloads never proposes a spotify_id already present in already_local for the
# queue that wants it.
# touches: collecter.download_tracking.plan_downloads

# ids_to_download from a single plan_downloads call are deduplicated - a spotify_id
# wanted by multiple queues appears at most once in ids_to_download.
# touches: collecter.download_tracking.plan_downloads

# targets_for a spotify_id wanted by N queues returns exactly those N queue names,
# no dupes, no drops, regardless of dict ordering.
# touches: collecter.download_tracking.targets_for

# resolve_downloads is idempotent - clearing an id already absent from in_flight is a
# no-op, and resolve_downloads(resolve_downloads(state, ids), ids) == resolve_downloads(state, ids).
# touches: collecter.download_tracking.resolve_downloads, collecter.download_tracking.DownloadState

# plan_cleanup never marks a file for deletion if its filename contains any protected id
# (in_flight OR still-queued-locally) - "downloaded files are never deleted while still
# being processed by any embedder". Use a filename constructed as a deliberate substring
# collision (e.g. protected id "123" showing up inside unrelated id "1234") to make sure
# this is a real containment check, not accidental string equality.
# touches: collecter.download_tracking.plan_cleanup

# plan_cleanup does mark a file for deletion when its filename contains no protected id -
# the inverse of the above, needed so the check isn't trivially satisfied by
# "delete nothing ever".
# touches: collecter.download_tracking.plan_cleanup

# match_on_disk never matches a filename to a spotify_id it doesn't actually contain
# (no false positives from substring collisions between different ids in wanted_ids) -
# same deliberate near-collision fixture as plan_cleanup's above.
# touches: collecter.download_tracking.match_on_disk

# A DownloadState round-tripped through plan_downloads then resolve_downloads for the
# same ids returns to having none of those ids in_flight (full tick cycle consistency).
# touches: collecter.download_tracking.plan_downloads, collecter.download_tracking.resolve_downloads
