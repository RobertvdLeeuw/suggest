# Shared fixture inventory for the whole suite.
#
# Scope reminder: unit/ should never depend on the DB fixtures - only
# integration/ (and the multiprocess test in integration/test_embedding_pipeline.py)
# needs a real Postgres.

import os
import uuid

import pytest
import pytest_asyncio
from pytest_postgresql import factories
from pytest_postgresql.janitor import DatabaseJanitor
from sqlalchemy.ext.asyncio import create_async_engine

import src.db as db
from mocks.clients import (
    FakeDownloaderClient,
    FakeLastFMClient,
    FakeMusicBrainzClient,
    FakeSpotifyClient,
)
from mocks.repository import FakeRepository
from src.models import Base

# --- DB fixtures (integration/ only) ----------------------------------------

# postgresql_proc: single throwaway PostgreSQL instance for the whole test
# session - pytest_postgresql spins this up once (session-scoped by default)
# and tears it down at the end of the run. Individual tests never talk to it
# directly; isolated_test_db below carves a fresh, uniquely-named database
# out of it per test function.
postgresql_proc = factories.postgresql_proc()


@pytest_asyncio.fixture
async def isolated_test_db(postgresql_proc):
    """Fresh, uniquely-named DB per test function, schema created via
    models.Base (NOT db.DatabaseManager.setup_tables()'s real Alembic path -
    that runs actual migration files against actual prod/dev DBs, which is
    the wrong tool for a throwaway per-test schema; Base.metadata.create_all
    is the lighter-weight equivalent tests want).

    db.DatabaseManager is a process-local singleton keyed by PID
    (db.py's `_instances: Dict[int, DatabaseManager]`) that only reads
    TEST_DATABASE_NAME/POSTGRES_*/DB_PORT from the environment on its FIRST
    initialize() call for this process - so a second test in the same
    process reusing the same singleton would silently keep pointing at the
    first test's database unless we force it to forget. That's why every
    test that gets this fixture ends with cleanup_all_instances(): it's not
    optional teardown hygiene here, it's what makes per-test env var
    overrides actually take effect on the NEXT test.

    Sets/restores TEST_DATABASE_NAME, POSTGRES_HOST/PASSWORD/USER, DB_PORT
    around the test so db.get_session() (called from repository.py's
    callers, or directly by db_session below) picks up this test's DB
    instead of whatever real .env config is present.
    """
    db_name = f"test_{uuid.uuid4().hex[:12]}"

    janitor = DatabaseJanitor(
        user=postgresql_proc.user,
        host=postgresql_proc.host,
        port=postgresql_proc.port,
        dbname=db_name,
        version=postgresql_proc.version,
        password=postgresql_proc.password,
    )
    janitor.init()

    env_overrides = {
        "TEST_DATABASE_NAME": db_name,
        "POSTGRES_HOST": postgresql_proc.host,
        "DB_PORT": str(postgresql_proc.port),
        "POSTGRES_USER": postgresql_proc.user,
        "POSTGRES_PASSWORD": postgresql_proc.password or "",
    }
    previous_env = {key: os.environ.get(key) for key in env_overrides}
    os.environ.update(env_overrides)

    try:
        engine = await db.get_db_manager().get_engine()

        async with engine.begin() as conn:
            await conn.execute(db.text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)

        yield db_name
    finally:
        await db.DatabaseManager.cleanup_all_instances()

        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        janitor.drop()


@pytest_asyncio.fixture
async def db_session(isolated_test_db):
    """AsyncSession bound to isolated_test_db's engine, via db.get_session()
    (the same context manager every real caller - services.py, main.py,
    embedders/__init__.py - uses) rather than constructing a session
    directly, so tests exercise the real session lifecycle (commit-on-exit,
    rollback-on-exception) and not a parallel one invented just for tests.

    NOTE: db.get_session()'s commit happens when THIS fixture tears down
    (i.e. after the test body returns), not when the test calls
    session.commit() itself - SqlAlchemyRepository methods already commit
    per-unit internally via _run_transactional, so this fixture's own
    commit-on-teardown is a no-op in the common case, not a substitute for
    it.
    """
    async with db.get_session() as session:
        yield session


@pytest.fixture
def test_db(isolated_test_db):
    """Convenience fixture exposing just the db_name from isolated_test_db,
    for the rare test that wants to assert against the database name itself
    (e.g. a DatabaseManager.create_database_url test) without pulling in a
    session."""
    return isolated_test_db


# --- Fake-backed fixtures (unit/ + integration/ orchestration tests) -------


@pytest.fixture
def fake_repo() -> FakeRepository:
    """Fresh mocks.repository.FakeRepository() per test, real in-memory
    get_or_create-style dedup semantics (not '...' stubs) - services.py's
    repo-lookup-before-resolve tests depend on the second lookup actually
    hitting."""
    return FakeRepository()


@pytest.fixture
def fake_clients() -> tuple[FakeSpotifyClient, FakeMusicBrainzClient, FakeLastFMClient]:
    """A fresh (spotify, musicbrainz, lastfm) fake triple, since most
    resolution.py/services.py tests need all three even when only one is
    under test. Returned as a plain tuple rather than three separate
    fixtures - tests destructure it (`spotify, musicbrainz, lastfm =
    fake_clients`), which keeps the common case a one-line pull instead of
    three fixture params for tests that touch all three anyway."""
    return FakeSpotifyClient(), FakeMusicBrainzClient(), FakeLastFMClient()


@pytest.fixture
def fake_downloader() -> FakeDownloaderClient:
    """Fresh mocks.clients.FakeDownloaderClient() per test."""
    return FakeDownloaderClient()


# --- Filesystem fixtures -----------------------------------------------------


@pytest.fixture
def tmp_download_dir(tmp_path) -> str:
    """Real empty directory standing in for download_dir - needed anywhere
    touching match_on_disk/plan_cleanup/download_loop's os.listdir, without
    ever writing outside pytest's tmp tree. Returned as a str (not a Path):
    every real call site (download.py, download_tracking.py) takes
    download_dir as a plain string and does f"{download_dir}/{fname}"
    string formatting, not Path joins - matching that exactly means a test
    building an expected path the same way the source does won't quietly
    diverge on a Path-vs-str formatting difference."""
    return str(tmp_path)


# --- Markers -----------------------------------------------------------------
#
# Registered in pytest.ini (not here) - see that file's comment for the
# slow/collecter/integration contract. pytest.ini is also where the
# pythonpath convention (src.collecter.../src.models resolving from repo
# root) lives, alongside asyncio_mode = auto for pytest-asyncio.
