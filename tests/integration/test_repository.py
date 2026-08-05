# Database sessions always commit or rollback cleanly (never left in intermediate state).
# touches: collecter.repository.SqlAlchemyRepository._run_transactional, conftest.db_session

# All inserts that don't follow constraints are failed gracefully.
# This one needs to be fleshed way the fuck out.
# touches: collecter.repository.get_or_create, collecter.repository._classify_db_error,
#          conftest.db_session

# Database connection pool behaves sanely when demand exceeds pool_size+max_overflow -
# extra callers queue for a connection (or get pool_timeout's error), they don't hang
# forever or crash the process. DECIDED: 1-2 fixed example-based tests (spin up more
# concurrent sessions than the pool allows, assert queueing/timeout behavior, not a
# hang) - not hypothesis-fuzzed. Not worth a separate load/chaos-test category for a
# single-operator project; this stays here.
# touches: src.db (engine/pool configuration), conftest.db_session

# Constraint violations during bulk operations don't corrupt successfully inserted data
# (test partial failure scenarios in your batch operations).
# touches: collecter.repository.get_or_create_many, conftest.db_session

# get_or_create is race-safe, for every model that uses it (Artist, Song, User,
# ArtistMetadata, SongMetadata via get_or_create_many): every time 2 processes try to
# create the same object concurrently, their contents match, only 1 instance is pushed
# to db, and both return a valid reference to that single instance.
# Parametrize across models instead of one-off tests per entity.
# touches: collecter.repository.get_or_create, collecter.repository.get_or_create_many,
#          collecter.repository.SqlAlchemyRepository.create_artist/create_song/get_or_create_user,
#          conftest.db_session

# get_or_create_many is race-safe for batch inserts specifically, not just single-row
# get_or_create - concurrent overlapping batches never produce duplicate rows.
# touches: collecter.repository.get_or_create_many, conftest.db_session

# _run_transactional retries the entire unit on serialization_failure/deadlock_detected
# sqlstates (40001/40P01) specifically, and does NOT retry on any other IntegrityError -
# a real constraint violation fails once, doesn't loop.
# touches: collecter.repository.SqlAlchemyRepository._run_transactional,
#          collecter.repository._classify_db_error, conftest.db_session

# System recovers gracefully from a dropped DB connection mid-transaction (rolls back
# cleanly, a subsequent call succeeds once the connection's back) - not a full outage
# simulation harness, just "kill the connection once, confirm no corruption and no
# permanently-wedged session." DECIDED: 1 example-based test, same reasoning as the
# pool-exhaustion assertion above.
# touches: collecter.repository.SqlAlchemyRepository._run_transactional, conftest.db_session

# enqueue_tracks never enqueues a spotify_id for an embedder that already has
# embeddings for it (per its own docstring contract) - verified against a real DB
# with pre-existing EmbeddingJukeMIR/EmbeddingAuditus rows.
# touches: collecter.repository.SqlAlchemyRepository.enqueue_tracks, conftest.db_session

# mark_/clear_artist_metadata_pending and mark_/clear_song_metadata_pending are
# idempotent against a real DB - double-mark produces no duplicate PendingArtistMetadata/
# PendingSongMetadata rows, double-clear is a harmless no-op.
# touches: collecter.repository.SqlAlchemyRepository (mark_/clear_*_metadata_pending),
#          conftest.db_session

# ListenChunks always uphold DB constraints.
# touches: collecter.repository.SqlAlchemyRepository.add_listen, models.ListenChunk,
#          conftest.db_session
