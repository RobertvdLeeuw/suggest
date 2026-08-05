"""
In-memory implementation of the Repository Protocol - dict-backed, no DB.
Lets services.py orchestration be tested (e.g. lookup-before-resolve
ordering, unavailable-source bookkeeping) without a real Postgres instance.

The real race-condition/transactional guarantees of get_or_create are NOT
re-tested here - that's what integration/test_repository.py + a real DB is
for. This fake's job is correct *sequential* semantics only.
"""

# FakeRepository: implements the Repository Protocol (collecter.repository.Repository).
#
# needs real (not '...' stub) in-memory behavior for:
#   - get_artist_by_spotify_id / create_artist: second lookup after a create
#     must return the same object - services.py's push_artist tests depend on
#     this actually short-circuiting the resolve call on a repeat push.
#   - get_song_by_spotify_id / create_song: same repeat-lookup requirement.
#   - get_or_create_user
#   - add_listen: needs to store ListenChunks alongside the Listen, since
#     listen_tracking-adjacent tests check chunk persistence.
#   - enqueue_tracks / dequeue_track / get_queued_track_ids: real queue-table
#     semantics per embedder (JukeMIR/Auditus separately) - enqueue must skip
#     ids already embedded for that embedder, matching the real
#     SqlAlchemyRepository.enqueue_tracks docstring's contract.
#   - save_embeddings / is_song_embedded
#   - get_random_artists
#   - mark_/clear_artist_metadata_pending, get_stale_pending_artists
#   - mark_/clear_song_metadata_pending, get_stale_pending_songs
#     (idempotent mark/clear - double-mark and double-clear must be no-ops,
#     not errors, matching the real repository's documented contract)
#
# needs a .calls log (method name + args) same as mocks/clients.py's fakes,
# for services.py's ordering assertions.
#
# touches: collecter.repository.Repository (the Protocol),
#          models.Artist, Song, User, Listen, ListenChunk, ArtistMetadata,
#          SongMetadata, PendingArtistMetadata, PendingSongMetadata,
#          QueueJukeMIR, QueueAuditus, EmbeddingJukeMIR, EmbeddingAuditus
