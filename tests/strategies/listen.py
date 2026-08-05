"""
Strategies for listen_tracking.py's process_playback_tick - a pure state
machine, so what actually matters here is generating realistic *sequences*
of playback snapshots (play/pause/skip/restart/rewind/track-change
patterns), not just one-off snapshots. A single random snapshot barely
exercises the state machine; a sequence does.
"""

# playback_snapshot_strat: single snapshot dict (item.id, item.duration_ms,
# progress_ms, is_playing, item.type) - reuses/wraps
# strategies.apis.spotify_playback_snapshot_strat rather than redefining the
# shape, see that file's note on this.
# touches: strategies.apis.spotify_playback_snapshot_strat

# snapshot_sequence_strat: a list of playback_snapshot_strat() results shaped
# into one of a few named realistic patterns rather than pure iid randomness -
# needs explicit generators (or a sampled_from over generator functions) for
# at least:
#   - normal playthrough (progress_ms monotonically increasing on one track,
#     then a track change)
#   - restart (progress_ms drops below duration_ms * 0.1)
#   - rewind (progress_ms drops but not below the near-start threshold)
#   - fast-forward / skip (progress_ms jumps forward by more than
#     sleep_time_s * 5000)
#   - pause/resume (is_playing toggling without item changing)
#   - track skipped via queue (new track's id matches next_in_queue_id)
#   - track completed naturally (ms_played reaches >= duration_ms * 0.75)
# each pattern should be checkable against process_playback_tick's actual
# branching logic (_classify_conclusion's four cases) so every branch has at
# least one generator that reliably hits it, not just hoped-for coverage
# from pure randomness.
# touches: collecter.listen_tracking.process_playback_tick,
#          collecter.listen_tracking.TrackingState,
#          collecter.listen_tracking._classify_conclusion (for reference on
#          what each pattern needs to trigger, not to be called directly)

# next_in_queue_id_strat: matches or doesn't match the current track id,
# needed to drive the "skipped via queue" _classify_conclusion branch
# specifically.
