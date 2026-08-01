# plan_downloads never proposes a spotify_id that match_on_disk already found on disk -
# it goes into queue_puts instead of ids_to_download.
# touches: collecter.download_tracking.plan_downloads, collecter.download_tracking.match_on_disk,
#          strategies.download.wanted_by_queue_strat, strategies.download.on_disk_strat

# plan_downloads never re-proposes a spotify_id already in state.in_flight from a
# previous tick.
# touches: collecter.download_tracking.plan_downloads, collecter.download_tracking.DownloadState,
#          strategies.download.download_state_strat

# plan_downloads never proposes a spotify_id already present in already_local for the
# queue that wants it.
# touches: collecter.download_tracking.plan_downloads, strategies.download.already_local_strat

# ids_to_download from a single plan_downloads call are deduplicated - a spotify_id
# wanted by multiple queues appears at most once in ids_to_download.
# touches: collecter.download_tracking.plan_downloads

# targets_for a spotify_id wanted by N queues returns exactly those N queue names,
# no dupes, no drops, regardless of dict ordering.
# touches: collecter.download_tracking.targets_for, strategies.download.wanted_by_queue_strat

# resolve_downloads is idempotent - clearing an id already absent from in_flight is a
# no-op, and resolve_downloads(resolve_downloads(state, ids), ids) == resolve_downloads(state, ids).
# touches: collecter.download_tracking.resolve_downloads, collecter.download_tracking.DownloadState

# plan_cleanup never marks a file for deletion if its filename contains any protected id
# (in_flight OR still-queued-locally) - "downloaded files are never deleted while still
# being processed by any embedder".
# touches: collecter.download_tracking.plan_cleanup, strategies.download.protected_and_disk_files_strat

# plan_cleanup does mark a file for deletion when its filename contains no protected id -
# the inverse of the above, needed so the property isn't trivially satisfied by
# "delete nothing ever".
# touches: collecter.download_tracking.plan_cleanup, strategies.download.protected_and_disk_files_strat

# match_on_disk never matches a filename to a spotify_id it doesn't actually contain
# (no false positives from substring collisions between different ids in wanted_ids).
# touches: collecter.download_tracking.match_on_disk, strategies.download.on_disk_strat

# A DownloadState round-tripped through plan_downloads then resolve_downloads for the
# same ids returns to having none of those ids in_flight (full tick cycle consistency).
# touches: collecter.download_tracking.plan_downloads, collecter.download_tracking.resolve_downloads
