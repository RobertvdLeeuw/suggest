"""
Purpose & scope
---------------
The ONLY file in src/collecter allowed to import sqlalchemy or hold an
AsyncSession. Every DB write in the collector goes through here, and every
write goes through the single get_or_create()/get_or_create_many() helpers
below - no entity gets its own hand-rolled insert/IntegrityError-catch/
re-select dance. That's the fix for the race conditions on Artist/Song/User
creation: one code path, gotten right once, covered by
tests/integration/test_repository.py.

get_or_create() only covers the conflict on the constraint you point it at.
It does NOT cover: a concurrent transaction hitting a *different* constraint
mid-unit-of-work (which poisons the whole Postgres transaction, not just the
one statement), deadlocks between two concurrent multi-row writes (e.g. two
overlapping enqueue_tracks calls), or a plain connection blip. All three are
retryable at the transaction level, not the statement level - that's what
SqlAlchemyRepository._run_transactional()/_classify_db_error() are for: every
public method below runs its body as one _run_transactional() unit, which
commits on success and rolls back + reruns the *entire* unit on a retryable
DB error. Reruns are safe because every write inside a unit goes through
get_or_create/get_or_create_many, both idempotent by construction.

Rules for this file:
  - Never import resolution.py or any client Protocol - this file only knows
    about ORM rows going in and out, not where they came from.
  - `Repository` is the Protocol services.py depends on; `SqlAlchemyRepository`
    is the only real implementation. tests/mocks/repository.py provides an
    in-memory fake of the same Protocol for tests that don't need a real DB.
  - Every public SqlAlchemyRepository method wraps its body in
    self._run_transactional(...) and does not call session.commit()/rollback()
    itself - that's _run_transactional's job, in exactly one place.
"""

import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable, Protocol, TypeVar

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from models import (
    Artist,
    ArtistMetadata,
    Listen,
    ListenChunk,
    PendingArtistMetadata,
    PendingSongMetadata,
    Song,
    SongArtist,
    SongMetadata,
    User,
)

from .clients.retry import with_backoff

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class Repository(Protocol):
    async def get_artist_by_spotify_id(self, spotify_id: str) -> Artist | None: ...
    async def create_artist(self, artist: Artist, metadata: list[ArtistMetadata]) -> Artist: ...

    async def get_song_by_spotify_id(self, spotify_id: str) -> Song | None: ...
    async def create_song(
        self, song: Song, artists: list[Artist], metadata: list[SongMetadata]
    ) -> Song: ...

    async def get_or_create_user(self, spotify_id: str, username: str) -> User: ...

    async def add_listen(self, user_id: int, song_id: int, listen_data: dict) -> Listen: ...

    async def enqueue_tracks(self, spotify_track_ids: list[str]) -> None:
        """Adds tracks to the JukeMIR/Auditus embedding queues, skipping any already
        queued or already embedded."""
        ...

    async def get_random_artists(self, n: int) -> list[Artist]: ...

    async def mark_artist_metadata_pending(self, artist_id: int, sources: set[str]) -> None: ...
    async def clear_artist_metadata_pending(self, artist_id: int, sources: set[str]) -> None: ...
    async def get_stale_pending_artists(self, older_than: datetime) -> list[Artist]: ...

    async def mark_song_metadata_pending(self, song_id: int, sources: set[str]) -> None: ...
    async def clear_song_metadata_pending(self, song_id: int, sources: set[str]) -> None: ...
    async def get_stale_pending_songs(self, older_than: datetime) -> list[Song]: ...


def _classify_db_error(exc: Exception) -> tuple[str, float] | str:
    """Serialization failures and deadlocks are Postgres telling us to retry the
    whole transaction from scratch, not a real problem with the data or a bug -
    they're expected under concurrent pushes of the same artist/song. A plain
    OperationalError (connection blip) is retryable too. Anything else (a real
    constraint violation, bad SQL, a programming error) is fatal - retrying it
    would just fail the same way every time."""
    code = getattr(getattr(exc, "orig", None), "sqlstate", None)
    if code in ("40001", "40P01"):  # serialization_failure, deadlock_detected
        return "retry", 0.0
    if isinstance(exc, OperationalError):
        return "retry"
    return "fatal"


class SqlAlchemyRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def _run_transactional(self, unit: Callable[[], Awaitable[T]], max_tries: int = 3) -> T:
        """Runs `unit` (which does its own get_or_create/get_or_create_many calls but
        never commits or rolls back itself), commits on success, and on a retryable
        DB error rolls back + reruns `unit` from scratch via clients/retry.py's
        with_backoff. Every public method on this class calls this exactly once, at
        the top level - `unit` should never call it again internally."""

        async def _attempt():
            try:
                result = await unit()
                await self._session.commit()
                return result
            except Exception:
                await self._session.rollback()
                raise

        return await with_backoff(_attempt, _classify_db_error, max_tries=max_tries)

    # --- Artist -------------------------------------------------------------

    async def get_artist_by_spotify_id(self, spotify_id: str) -> Artist | None:
        result = await self._session.execute(select(Artist).where(Artist.spotify_id == spotify_id))
        return result.scalar_one_or_none()

    async def create_artist(self, artist: Artist, metadata: list[ArtistMetadata]) -> Artist:
        async def _unit() -> Artist:
            row, _ = await get_or_create(
                self._session,
                Artist,
                unique_cols={"spotify_id": artist.spotify_id},
                defaults={"artist_name": artist.artist_name},
            )

            if metadata:
                await get_or_create_many(
                    self._session,
                    ArtistMetadata,
                    rows=[
                        {
                            "artist_id": row.artist_id,
                            "type": m.type,
                            "value": m.value,
                            "source": m.source,
                        }
                        for m in metadata
                    ],
                    unique_cols=("artist_id", "type", "value", "source"),
                )
            return row

        return await self._run_transactional(_unit)

    # --- Song -----------------------------------------------------------------

    async def get_song_by_spotify_id(self, spotify_id: str) -> Song | None:
        result = await self._session.execute(select(Song).where(Song.spotify_id == spotify_id))
        return result.scalar_one_or_none()

    async def create_song(
        self, song: Song, artists: list[Artist], metadata: list[SongMetadata]
    ) -> Song:
        """`artists` must already be persisted (created via create_artist first) -
        this only links them, it never creates an Artist row itself."""

        async def _unit() -> Song:
            row, _ = await get_or_create(
                self._session,
                Song,
                unique_cols={"spotify_id": song.spotify_id},
                defaults={"song_name": song.song_name},
            )

            if artists:
                await get_or_create_many(
                    self._session,
                    SongArtist,
                    rows=[
                        {"song_id": row.song_id, "artist_id": artist.artist_id}
                        for artist in artists
                    ],
                    unique_cols=("song_id", "artist_id"),
                )

            if metadata:
                await get_or_create_many(
                    self._session,
                    SongMetadata,
                    rows=[
                        {
                            "song_id": row.song_id,
                            "type": m.type,
                            "value": m.value,
                            "source": m.source,
                        }
                        for m in metadata
                    ],
                    unique_cols=("song_id", "type", "value", "source"),
                )
            return row

        return await self._run_transactional(_unit)

    # --- User -----------------------------------------------------------------

    async def get_or_create_user(self, spotify_id: str, username: str) -> User:
        async def _unit() -> User:
            row, _ = await get_or_create(
                self._session,
                User,
                unique_cols={"spotify_id": spotify_id},
                defaults={"username": username},
            )
            return row

        return await self._run_transactional(_unit)

    # --- Listens ----------------------------------------------------------

    async def add_listen(self, user_id: int, song_id: int, listen_data: dict) -> Listen:
        """listen_data holds everything from listen_tracking.ListenEvent except
        spotify_id: ms_played, reason_start, reason_end, chunks (list of
        {"from_ms", "to_ms"} dicts), and optionally from_history."""
        chunks = listen_data.pop("chunks", [])
        listened_at = listen_data.pop("listened_at", None) or datetime.now(timezone.utc)

        async def _unit() -> Listen:
            row, _ = await get_or_create(
                self._session,
                Listen,
                unique_cols={
                    "user_id": user_id,
                    "song_id": song_id,
                    "listened_at": listened_at,
                },
                defaults=listen_data,
            )

            if chunks:
                await get_or_create_many(
                    self._session,
                    ListenChunk,
                    rows=[
                        {"listen_id": row.listen_id, "from_ms": c["from_ms"], "to_ms": c["to_ms"]}
                        for c in chunks
                    ],
                    unique_cols=("listen_id", "from_ms", "to_ms"),
                )
            return row

        return await self._run_transactional(_unit)

    # --- Embedding queues -------------------------------------------------

    async def enqueue_tracks(self, spotify_track_ids: list[str]) -> None:
        """Queues tracks for JukeMIR and/or Auditus embedding, per embedder:
        deduped against rows already in that embedder's queue table (ON CONFLICT
        DO NOTHING on spotify_id, its primary key), AND skipped entirely for that
        embedder if the track already has embeddings there. The latter is what
        keeps start_download_loop (old/downloader.py) - which reads candidates
        straight off QueueJukeMIR/QueueAuditus - from ever seeing an
        already-embedded track, so it never re-downloads audio for it.

        A track with no Song row yet (never pushed) obviously has no embeddings
        either, and queues normally for both. A track embedded by one embedder
        but not the other still queues for the one that's missing."""
        from models import EmbeddingAuditus, EmbeddingJukeMIR, QueueAuditus, QueueJukeMIR

        if not spotify_track_ids:
            return

        async def _unit() -> None:
            song_ids_by_spotify_id = dict(
                (
                    await self._session.execute(
                        select(Song.spotify_id, Song.song_id).where(
                            Song.spotify_id.in_(spotify_track_ids)
                        )
                    )
                ).all()
            )
            known_song_ids = list(song_ids_by_spotify_id.values())

            for queue_model, embedding_model in (
                (QueueJukeMIR, EmbeddingJukeMIR),
                (QueueAuditus, EmbeddingAuditus),
            ):
                already_embedded_song_ids = set()
                if known_song_ids:
                    result = await self._session.execute(
                        select(embedding_model.song_id)
                        .where(embedding_model.song_id.in_(known_song_ids))
                        .distinct()
                    )
                    already_embedded_song_ids = {row[0] for row in result.all()}

                to_queue = [
                    sid
                    for sid in spotify_track_ids
                    if song_ids_by_spotify_id.get(sid) not in already_embedded_song_ids
                ]

                if to_queue:
                    await get_or_create_many(
                        self._session,
                        queue_model,
                        rows=[{"spotify_id": sid} for sid in to_queue],
                        unique_cols=("spotify_id",),
                    )

        await self._run_transactional(_unit)

    # --- Random sampling (queue_similar_artists) ---------------------------

    async def get_random_artists(self, n: int) -> list[Artist]:
        result = await self._session.execute(select(Artist).order_by(func.random()).limit(n))
        return list(result.scalars().all())

    # --- Pending metadata (resolution sources marked UNAVAILABLE) ---------
    #
    # mark_*/clear_* both go through get_or_create_many/a plain DELETE rather
    # than an upsert - re-marking an already-pending source is a deliberate
    # no-op (see PendingArtistMetadata's docstring in models.py), and clearing
    # a source that was never pending is just as harmless a no-op. Neither
    # needs _run_transactional's retry-the-whole-unit machinery: both are a
    # single statement, and get_or_create_many is already race-safe on its own.

    async def mark_artist_metadata_pending(self, artist_id: int, sources: set[str]) -> None:
        if not sources:
            return

        async def _unit() -> None:
            await get_or_create_many(
                self._session,
                PendingArtistMetadata,
                rows=[{"artist_id": artist_id, "source": s} for s in sources],
                unique_cols=("artist_id", "source"),
            )

        await self._run_transactional(_unit)

    async def clear_artist_metadata_pending(self, artist_id: int, sources: set[str]) -> None:
        if not sources:
            return

        async def _unit() -> None:
            await self._session.execute(
                delete(PendingArtistMetadata).where(
                    PendingArtistMetadata.artist_id == artist_id,
                    PendingArtistMetadata.source.in_(sources),
                )
            )

        await self._run_transactional(_unit)

    async def get_stale_pending_artists(self, older_than: datetime) -> list[Artist]:
        result = await self._session.execute(
            select(Artist)
            .join(PendingArtistMetadata, PendingArtistMetadata.artist_id == Artist.artist_id)
            .where(PendingArtistMetadata.created_at < older_than)
            .distinct()
        )
        return list(result.scalars().all())

    async def mark_song_metadata_pending(self, song_id: int, sources: set[str]) -> None:
        if not sources:
            return

        async def _unit() -> None:
            await get_or_create_many(
                self._session,
                PendingSongMetadata,
                rows=[{"song_id": song_id, "source": s} for s in sources],
                unique_cols=("song_id", "source"),
            )

        await self._run_transactional(_unit)

    async def clear_song_metadata_pending(self, song_id: int, sources: set[str]) -> None:
        if not sources:
            return

        async def _unit() -> None:
            await self._session.execute(
                delete(PendingSongMetadata).where(
                    PendingSongMetadata.song_id == song_id,
                    PendingSongMetadata.source.in_(sources),
                )
            )

        await self._run_transactional(_unit)

    async def get_stale_pending_songs(self, older_than: datetime) -> list[Song]:
        result = await self._session.execute(
            select(Song)
            .join(PendingSongMetadata, PendingSongMetadata.song_id == Song.song_id)
            .where(PendingSongMetadata.created_at < older_than)
            .distinct()
        )
        return list(result.scalars().all())


async def get_or_create(
    session: AsyncSession,
    model: type[T],
    unique_cols: dict,
    defaults: dict | None = None,
) -> tuple[T, bool]:
    """Race-safe get-or-create via INSERT ... ON CONFLICT DO NOTHING + re-select.
    Returns (row, created). Used by every SqlAlchemyRepository method that creates
    an entity - Artist, Song, User, and metadata rows all go through this, not
    a bespoke version each.

    `unique_cols` must exactly match the columns of a real unique/PK constraint on
    `model` - this is what ON CONFLICT targets. A dict that doesn't match one won't
    raise here; it'll just fail to suppress the conflict, surfacing as an
    IntegrityError from the INSERT instead (which _run_transactional will then
    treat as fatal, correctly, since that's a real bug, not a race)."""
    stmt = (
        pg_insert(model)
        .values(**unique_cols, **(defaults or {}))
        .on_conflict_do_nothing(index_elements=list(unique_cols))
        .returning(model)
    )
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()
    if row is not None:
        return row, True

    # Someone else won the race (or the row already existed) - re-select rather
    # than retry the insert, since retrying would just conflict again.
    result = await session.execute(select(model).filter_by(**unique_cols))
    return result.scalar_one(), False


async def get_or_create_many(
    session: AsyncSession,
    model: type[T],
    rows: list[dict],
    unique_cols: tuple[str, ...],
) -> list[T]:
    """Bulk variant of get_or_create for metadata rows (ArtistMetadata/SongMetadata)
    and link/queue rows (SongArtist, QueueJukeMIR/QueueAuditus), which need the same
    race-safety as their parent entity - a concurrent resolve of the same artist
    shouldn't be able to violate uq_artist_metadata either.

    One INSERT ... ON CONFLICT DO NOTHING ... RETURNING for the whole batch, then one
    re-select for whichever rows didn't come back (already existed, or lost a
    concurrent race). Order relative to `rows` is not preserved."""
    if not rows:
        return []

    stmt = (
        pg_insert(model)
        .values(rows)
        .on_conflict_do_nothing(index_elements=list(unique_cols))
        .returning(model)
    )
    result = await session.execute(stmt)
    created = list(result.scalars().all())

    if len(created) == len(rows):
        return created

    created_keys = {tuple(getattr(r, col) for col in unique_cols) for r in created}
    missing = [row for row in rows if tuple(row[col] for col in unique_cols) not in created_keys]

    # missing rows already existed (or lost a race) - re-select them by their
    # unique-column values, one OR'd query rather than one SELECT per row.
    # Group per-row so each row's columns are AND'd together, rows OR'd together.
    row_conditions = [
        and_(*(getattr(model, col) == row[col] for col in unique_cols)) for row in missing
    ]
    result = await session.execute(select(model).where(or_(*row_conditions)))
    return created + list(result.scalars().all())
