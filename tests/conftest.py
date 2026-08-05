# Shared fixture inventory for the whole suite. Nothing here is implemented yet -
# each block below is one fixture, described the same way test stubs describe
# assertions: what it needs to do, then what it needs to touch.
#
# Scope reminder: unit/ should never depend on the DB fixtures - only
# integration/ (and the multiprocess test in integration/test_embedding_pipeline.py)
# needs a real Postgres.

# --- DB fixtures (integration/ only) ----------------------------------------

# postgresql_proc: single PostgreSQL instance for the whole test session.
# touches: pytest_postgresql.factories.postgresql_proc

# event_loop: session-scoped asyncio event loop.
# touches: asyncio.get_event_loop_policy

# isolated_test_db: fresh, uniquely-named DB per test function, schema created via
# models.Base, torn down after. Sets/restores TEST_MODE, TEST_DATABASE_NAME,
# POSTGRES_HOST/PORT/USER env vars around the test so db.get_session() picks it up.
# touches: pytest_postgresql.janitor.DatabaseJanitor, sqlalchemy.ext.asyncio.create_async_engine,
#          models.Base.metadata.create_all, db.DatabaseManager.cleanup_all_instances

# db_session: AsyncSession bound to isolated_test_db's engine.
# touches: sqlalchemy.ext.asyncio.async_sessionmaker, AsyncSession

# test_db: convenience fixture exposing just the db_name from isolated_test_db.

# --- Fake-backed fixtures (unit/ + integration/ orchestration tests) -------

# fake_repo: fresh mocks.repository.FakeRepository() per test, real in-memory
# get_or_create-style dedup semantics (not '...' stubs) - services.py's
# repo-lookup-before-resolve tests depend on the second lookup actually hitting.
# touches: mocks.repository.FakeRepository

# fake_clients: bundles a fresh (spotify, musicbrainz, lastfm) fake triple from
# mocks.clients, since most resolution.py/services.py tests need all three even
# when only one is under test.
# touches: mocks.clients.FakeSpotifyClient, FakeMusicBrainzClient, FakeLastFMClient

# fake_downloader: fresh mocks.clients.FakeDownloaderClient() per test.
# touches: mocks.clients.FakeDownloaderClient

# --- Filesystem fixtures -----------------------------------------------------

# tmp_download_dir: real empty directory (tmp_path-based) standing in for
# download_dir - needed anywhere touching match_on_disk/plan_cleanup/download_loop's
# os.listdir, without ever writing outside pytest's tmp tree.
# touches: pytest tmp_path

# --- Markers -----------------------------------------------------------------

# Register in pyproject.toml/pytest.ini, not here, but noting the contract:
#   slow        - real multiprocessing / long-running (integration/test_song_queue_concurrency.py,
#                 integration/test_embedding_pipeline.py)
#   collecter   - anything under this package, mirrors old pytestmark convention
#   integration - anything needing a real DB or real subprocesses
# unit/ should be runnable with none of these selected, fast, no external state.
