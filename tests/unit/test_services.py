# push_artist never calls resolution.resolve_artist when repo.get_artist_by_spotify_id
# already returns a hit - repo lookup always short-circuits resolution.
# touches: collecter.services.push_artist, mocks.repository.FakeRepository (pre-seeded),
#          mocks.clients.FakeSpotifyClient/FakeMusicBrainzClient/FakeLastFMClient (.calls log)

# push_track never calls resolution.resolve_track when repo.get_song_by_spotify_id
# already returns a hit - same short-circuit rule as push_artist.
# touches: collecter.services.push_track, mocks.repository.FakeRepository, mocks.clients (.calls log)

# push_track resolves/pushes every artist on the track via push_artist, in
# resolved.artists order, before creating the Song row.
# touches: collecter.services.push_track, collecter.services.push_artist, mocks.repository.FakeRepository

# A push_* call that hits an unavailable source still returns a created Artist/Song row
# (never raises), and records exactly the unavailable sources via
# repo.mark_artist_metadata_pending / repo.mark_song_metadata_pending - never blocks
# the write.
# touches: collecter.services.push_artist, collecter.services.push_track,
#          mocks.repository.FakeRepository, strategies.resolution.client_failure_sequence_strat

# retry_pending_metadata only re-resolves artists/songs from
# repo.get_stale_pending_artists/get_stale_pending_songs (older_than cutoff respected),
# and clears exactly the sources that succeeded this attempt via
# clear_artist_metadata_pending/clear_song_metadata_pending - sources still failing
# stay pending, not cleared.
# touches: collecter.services.retry_pending_metadata, mocks.repository.FakeRepository

# retry_pending_metadata re-creating an already-existing artist/song is a safe no-op on
# the row itself (repo.create_artist/create_song go through get_or_create semantics) -
# only pending-metadata bookkeeping changes.
# touches: collecter.services.retry_pending_metadata, mocks.repository.FakeRepository

# add_song_listens skips malformed track entries (missing spotify_id or ms_played)
# rather than raising.
# touches: collecter.services.add_song_listens

# add_history_listens correctly maps Spotify's raw history reason codes via
# _HISTORY_START_REASON_MAP/_HISTORY_END_REASON_MAP, and skips entries with no
# spotify_track_uri or with a spotify_episode_uri present (podcasts excluded).
# touches: collecter.services.add_history_listens, collecter.services._HISTORY_START_REASON_MAP,
#          collecter.services._HISTORY_END_REASON_MAP

# queue_new_tracks filters out None ids before calling repo.enqueue_tracks - never
# passes a None spotify_id through to the repository layer.
# touches: collecter.services.queue_new_tracks, mocks.repository.FakeRepository

# run_recent_listen_loop enqueues a newly-encountered track (repo.enqueue_tracks) on
# start/change of the current track id, but not on resume/restart of the same track.
# touches: collecter.services.run_recent_listen_loop, collecter.listen_tracking.process_playback_tick,
#          mocks.repository.FakeRepository, mocks.clients.FakeSpotifyClient,
#          strategies.listen.snapshot_sequence_strat
