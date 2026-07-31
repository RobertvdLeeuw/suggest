"""
Purpose & scope
---------------
The ONLY file in src/collecter allowed to import sqlalchemy or hold an
AsyncSession. Every DB write in the collector goes through here, and every
write goes through the single get_or_create() helper below - no entity gets
its own hand-rolled insert/IntegrityError-catch/re-select dance. That's the
fix for the race conditions on Artist/Song/User creation: one code path,
gotten right once, covered by tests/integration/test_repository.py.

Rules for this file:
  - Never import resolution.py or any client Protocol - this file only knows
    about ORM rows going in and out, not where they came from.
  - `Repository` is the Protocol services.py depends on; `SqlAlchemyRepository`
    is the only real implementation. tests/mocks/repository.py provides an
    in-memory fake of the same Protocol for tests that don't need a real DB.
"""

from typing import Protocol, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from models import Artist, Song, User, Listen, ArtistMetadata, SongMetadata

T = TypeVar("T")


class Repository(Protocol):
    async def get_artist_by_spotify_id(self, spotify_id: str) -> Artist | None: ...
    async def create_artist(self, artist: Artist, metadata: list[ArtistMetadata]) -> Artist: ...

    async def get_song_by_spotify_id(self, spotify_id: str) -> Song | None: ...
    async def create_song(self, song: Song, metadata: list[SongMetadata]) -> Song: ...

    async def get_or_create_user(self, spotify_id: str, username: str) -> User: ...

    async def add_listen(self, user_id: int, song_id: int, listen_data: dict) -> Listen: ...

    async def enqueue_tracks(self, spotify_track_ids: list[str]) -> None:
        """Adds tracks to the JukeMIR/Auditus embedding queues, skipping any already
        queued or already embedded."""
        ...


class SqlAlchemyRepository:
    def __init__(self, session: AsyncSession):
        ...

    async def get_artist_by_spotify_id(self, spotify_id: str) -> Artist | None: ...
    async def create_artist(self, artist: Artist, metadata: list[ArtistMetadata]) -> Artist: ...

    async def get_song_by_spotify_id(self, spotify_id: str) -> Song | None: ...
    async def create_song(self, song: Song, metadata: list[SongMetadata]) -> Song: ...

    async def get_or_create_user(self, spotify_id: str, username: str) -> User: ...

    async def add_listen(self, user_id: int, song_id: int, listen_data: dict) -> Listen: ...

    async def enqueue_tracks(self, spotify_track_ids: list[str]) -> None: ...


async def get_or_create(
    session: AsyncSession,
    model: type[T],
    unique_cols: dict,
    defaults: dict | None = None,
) -> tuple[T, bool]:
    """Race-safe get-or-create via INSERT ... ON CONFLICT DO NOTHING + re-select.
    Returns (row, created). Used by every SqlAlchemyRepository method that creates
    an entity - Artist, Song, User, and metadata rows all go through this, not
    a bespoke version each."""
    ...


async def get_or_create_many(
    session: AsyncSession,
    model: type[T],
    rows: list[dict],
    unique_cols: tuple[str, ...],
) -> list[T]:
    """Bulk variant of get_or_create for metadata rows (ArtistMetadata/SongMetadata),
    which need the same race-safety as their parent entity - a concurrent resolve of
    the same artist shouldn't be able to violate uq_artist_metadata either."""
    ...
