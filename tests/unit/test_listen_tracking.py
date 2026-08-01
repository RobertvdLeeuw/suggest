# ListenChunks of a listen add up to ROUGHLY (networking and floats) ms_played.
# touches: collecter.listen_tracking.process_playback_tick, collecter.listen_tracking.ListenEvent,
#          collecter.listen_tracking.ListenChunk, strategies.listen.snapshot_sequence_strat

# All listen start/end reasons for valid state transitions.
# touches: collecter.listen_tracking.process_playback_tick, collecter.listen_tracking._classify_conclusion,
#          strategies.listen.snapshot_sequence_strat (needs generators hitting every
#          _classify_conclusion branch: trackdone / skipped-via-queue / paused / unknown)

# No two consecutive "start" events for the same still-playing track - a tick that
# doesn't change the track and doesn't restart/rewind never emits a spurious event.
# touches: collecter.listen_tracking.process_playback_tick, collecter.listen_tracking.TrackingState

# Every emitted ListenEvent has non-negative ms_played, and every ListenChunk within it
# has from_ms <= to_ms (never a zero-or-negative-width chunk - _close_chunk already
# guards this, so this is confirming that guarantee holds end-to-end through a full
# tick sequence, not just the single-call unit level).
# touches: collecter.listen_tracking._close_chunk, collecter.listen_tracking.process_playback_tick

# Restart detection (progress_ms drops below near_start threshold) always closes the
# prior chunk and starts a fresh chunks=[] state - never carries stale chunks into the
# post-restart listen.
# touches: collecter.listen_tracking.process_playback_tick (the restart branch specifically)

# A next_in_queue_id change alone (queue reordering without a track change) never
# emits a ListenEvent by itself - only an actual track change does.
# touches: collecter.listen_tracking.process_playback_tick

# --- The following needs services.run_recent_listen_loop, not just process_playback_tick
# --- in isolation - overlaps test_services.py; kept here since the assertion is really
# --- about listen-event correctness under backoff, test_services.py owns pure ordering.

# An extra wait from backoff in recently played loop doesn't fuck up the listen item in any way.
# touches: collecter.services.run_recent_listen_loop, collecter.listen_tracking.process_playback_tick,
#          mocks.clients.FakeSpotifyClient, mocks.repository.FakeRepository

# "ListenChunks always uphold DB constraints" moved to integration/test_repository.py -
# constraint enforcement isn't observable against FakeRepository, needs a real DB.
