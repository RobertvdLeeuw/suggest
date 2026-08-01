# Failed downloads never make their way into the embedding tables - a handled
# DOWNLOAD_ERRORS failure in _download_one dequeues the id from every target queue's
# DB table (repo.dequeue_track) rather than leaving it queued or queuing it locally.
# touches: collecter.download._download_one, collecter.clients.downloader.DOWNLOAD_ERRORS,
#          mocks.clients.FakeDownloaderClient, mocks.repository.FakeRepository

# _download_one fans a single successful download out to every queue in target_names -
# one search()+download() call per spotify_id regardless of how many queues wanted it.
# touches: collecter.download._download_one, mocks.clients.FakeDownloaderClient (.calls log)

# An unhandled (non-DOWNLOAD_ERRORS) exception from _download_one propagates rather
# than being swallowed - download_loop's gather(..., return_exceptions=True) is what
# catches it, not _download_one itself.
# touches: collecter.download._download_one, collecter.download.download_loop

# download_loop only queries repo.get_queued_track_ids for a queue when
# len(queue) < QUEUE_MAX_LEN - a full local queue is skipped for that tick, not
# re-queried needlessly.
# touches: collecter.download.download_loop, collecter.embedders.song_queue.QUEUE_MAX_LEN,
#          mocks.repository.FakeRepository (.calls log)

# download_loop's queue_puts (files already found on disk) are applied to the correct
# local SongQueue by name before any actual download is dispatched for that tick.
# touches: collecter.download.download_loop, collecter.download_tracking.plan_downloads

# _clean_downloads only deletes files not protected by state.in_flight OR any queue's
# current local contents - never deletes a file another embedder still needs.
# touches: collecter.download._clean_downloads, collecter.download_tracking.plan_cleanup,
#          conftest.tmp_download_dir

# A renamed download (file_path -> {spotify_id}_{original_name}) only renames when the
# target path doesn't already exist - never silently overwrites/clobbers an existing file.
# touches: collecter.download._download_one, conftest.tmp_download_dir
