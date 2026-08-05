"""
In-memory implementation of the Repository Protocol - dict-backed, no DB.
Lets services.py orchestration be tested (e.g. lookup-before-resolve
ordering, unavailable-source bookkeeping) without a real Postgres instance.

The real race-condition/transactional guarantees of get_or_create are NOT
re-tested here - that's what integration/test_repository.py + a real DB is
for. This fake's job is correct *sequential* semantics only: dedup on
spotify_id, idempotent mark/clear, per-embedder queue fan-out, and so on -
whatever a single-threaded caller can observe by calling methods in order.

Design note on model types: Artist/Song/ArtistMetadata/SongMetadata rows are
never constructed here - callers (mapping.py, in practice) build real
`..models` instances and hand them to create_artist/create_song, so this
file only ever stores/mutates what it's given (setting the *_id once, the
same "DB assigns the primary key" moment get_or_create represents for
real). User/Listen/ListenChunk, by contrast, are this Protocol's to create -
rather than couple this fake to models.py's exact declarative-class
constructor signature (which this file's own tests don't exercise), it
builds small local stand-in rows with just the attributes callers are
documented to read (user.user_id, listen.song_id, listen.chunks, ...).
QueueJukeMIR/QueueAuditus/EmbeddingJukeMIR/EmbeddingAuditus ARE imported for
real, though - those are used purely as dict keys/type-identity, and
download.py/embedders code passes the real classes in (e.g. via
SongQueue.q_type), so this fake has to key off the same objects to be
usable in the same test as a real SongQueue.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.models import EmbeddingAuditus, EmbeddingJukeMIR, QueueAuditus, QueueJukeMIR

_EMBEDDER_PAIRS = ((QueueJukeMIR, EmbeddingJukeMIR), (QueueAuditus, EmbeddingAuditus))


@dataclass
class _FakeUser:
    user_id: int
    spotify_id: str
    username: str


@dataclass
class _FakeListenChunk:
    listen_id: int
    from_ms: int
    to_ms: int


@dataclass
class _FakeListen:
    listen_id: int
    user_id: int
    song_id: int
    listened_at: datetime
    ms_played: int | None = None
    reason_start: str | None = None
    reason_end: str | None = None
    from_history: bool = False
    chunks: list[_FakeListenChunk] = field(default_factory=list)


class FakeRepository:
    """Implements collecter.repository.Repository. See module docstring for
    what "real" means here vs. integration/test_repository.py's job.

    - self.calls: list[(method_name, args)] in call order, for services.py's
      ordering/short-circuit assertions (mirrors mocks/clients.py's fakes).
    """

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

        self._artists: dict[str, Any] = {}  # spotify_id -> Artist
        self._artists_by_id: dict[int, Any] = {}
        self._artist_metadata: dict[int, list[Any]] = {}
        self._next_artist_id = 1

        self._songs: dict[str, Any] = {}  # spotify_id -> Song
        self._songs_by_id: dict[int, Any] = {}
        self._song_metadata: dict[int, list[Any]] = {}
        self._song_artists: set[tuple[int, int]] = set()
        self._next_song_id = 1

        self._users: dict[str, _FakeUser] = {}  # spotify_id -> User
        self._next_user_id = 1

        self._listens_by_key: dict[tuple[int, int, datetime], _FakeListen] = {}
        self._next_listen_id = 1

        # Per-embedder queues/embeddings, keyed by the real QueueJukeMIR/
        # QueueAuditus/EmbeddingJukeMIR/EmbeddingAuditus classes themselves -
        # dict preserves insertion order, which is what stands in for
        # created_at ordering in get_queued_track_ids.
        self._queues: dict[type, dict[str, None]] = {}
        self._embedded_song_ids: dict[type, set[int]] = {}
        self._embeddings: dict[type, list[Any]] = {}

        self._pending_artist_metadata: dict[int, dict[str, datetime]] = {}
        self._pending_song_metadata: dict[int, dict[str, datetime]] = {}

    # --- Artist -------------------------------------------------------------

    async def get_artist_by_spotify_id(self, spotify_id: str):
        self.calls.append(("get_artist_by_spotify_id", (spotify_id,)))
        return self._artists.get(spotify_id)

    async def create_artist(self, artist, metadata: list):
        self.calls.append(("create_artist", (artist.spotify_id,)))

        existing = self._artists.get(artist.spotify_id)
        if existing is not None:
            row = existing
        else:
            artist.artist_id = self._next_artist_id
            self._next_artist_id += 1
            self._artists[artist.spotify_id] = artist
            self._artists_by_id[artist.artist_id] = artist
            row = artist

        stored = self._artist_metadata.setdefault(row.artist_id, [])
        seen = {(m.type, m.value, m.source) for m in stored}
        for m in metadata:
            key = (m.type, m.value, m.source)
            if key not in seen:
                stored.append(m)
                seen.add(key)

        return row

    # --- Song -----------------------------------------------------------------

    async def get_song_by_spotify_id(self, spotify_id: str):
        self.calls.append(("get_song_by_spotify_id", (spotify_id,)))
        return self._songs.get(spotify_id)

    async def create_song(self, song, artists: list, metadata: list):
        self.calls.append(("create_song", (song.spotify_id,)))

        existing = self._songs.get(song.spotify_id)
        if existing is not None:
            row = existing
        else:
            song.song_id = self._next_song_id
            self._next_song_id += 1
            self._songs[song.spotify_id] = song
            self._songs_by_id[song.song_id] = song
            row = song

        for artist in artists:
            self._song_artists.add((row.song_id, artist.artist_id))

        stored = self._song_metadata.setdefault(row.song_id, [])
        seen = {(m.type, m.value, m.source) for m in stored}
        for m in metadata:
            key = (m.type, m.value, m.source)
            if key not in seen:
                stored.append(m)
                seen.add(key)

        return row

    # --- User -----------------------------------------------------------------

    async def get_or_create_user(self, spotify_id: str, username: str) -> _FakeUser:
        self.calls.append(("get_or_create_user", (spotify_id, username)))

        existing = self._users.get(spotify_id)
        if existing is not None:
            return existing

        user = _FakeUser(user_id=self._next_user_id, spotify_id=spotify_id, username=username)
        self._next_user_id += 1
        self._users[spotify_id] = user
        return user

    # --- Listens ----------------------------------------------------------

    async def add_listen(self, user_id: int, song_id: int, listen_data: dict) -> _FakeListen:
        self.calls.append(("add_listen", (user_id, song_id)))

        listen_data = dict(listen_data)
        chunks = listen_data.pop("chunks", [])
        listened_at = listen_data.pop("listened_at", None) or datetime.now(timezone.utc)

        key = (user_id, song_id, listened_at)
        listen = self._listens_by_key.get(key)
        if listen is None:
            listen = _FakeListen(
                listen_id=self._next_listen_id,
                user_id=user_id,
                song_id=song_id,
                listened_at=listened_at,
                ms_played=listen_data.get("ms_played"),
                reason_start=listen_data.get("reason_start"),
                reason_end=listen_data.get("reason_end"),
                from_history=listen_data.get("from_history", False),
            )
            self._next_listen_id += 1
            self._listens_by_key[key] = listen

        seen = {(c.from_ms, c.to_ms) for c in listen.chunks}
        for c in chunks:
            ck = (c["from_ms"], c["to_ms"])
            if ck not in seen:
                listen.chunks.append(
                    _FakeListenChunk(
                        listen_id=listen.listen_id, from_ms=c["from_ms"], to_ms=c["to_ms"]
                    )
                )
                seen.add(ck)

        return listen

    # --- Embedding queues -------------------------------------------------

    async def enqueue_tracks(self, spotify_track_ids: list[str]) -> None:
        self.calls.append(("enqueue_tracks", tuple(spotify_track_ids)))

        if not spotify_track_ids:
            return

        song_id_by_spotify_id = {
            sid: self._songs[sid].song_id for sid in spotify_track_ids if sid in self._songs
        }

        for queue_model, embedding_model in _EMBEDDER_PAIRS:
            already_embedded = self._embedded_song_ids.get(embedding_model, set())
            to_queue = [
                sid
                for sid in spotify_track_ids
                if song_id_by_spotify_id.get(sid) not in already_embedded
            ]

            queue = self._queues.setdefault(queue_model, {})
            for sid in to_queue:
                queue.setdefault(sid, None)  # ON CONFLICT DO NOTHING on spotify_id

    async def save_embeddings(self, embeddings: list) -> None:
        self.calls.append(("save_embeddings", (len(embeddings),)))

        if not embeddings:
            return

        embedding_model = type(embeddings[0])
        self._embeddings.setdefault(embedding_model, []).extend(embeddings)
        self._embedded_song_ids.setdefault(embedding_model, set()).update(
            e.song_id for e in embeddings
        )

    async def is_song_embedded(self, song_id: int, embedding_model: type) -> bool:
        self.calls.append(("is_song_embedded", (song_id, embedding_model)))
        return song_id in self._embedded_song_ids.get(embedding_model, set())

    async def dequeue_track(self, queue_model: type, spotify_id: str) -> None:
        self.calls.append(("dequeue_track", (queue_model, spotify_id)))
        self._queues.get(queue_model, {}).pop(spotify_id, None)

    async def get_queued_track_ids(self, queue_model: type, limit: int) -> list[str]:
        self.calls.append(("get_queued_track_ids", (queue_model, limit)))
        if limit <= 0:
            return []
        return list(self._queues.get(queue_model, {}).keys())[:limit]

    # --- Random sampling (queue_similar_artists) ---------------------------

    async def get_random_artists(self, n: int) -> list:
        self.calls.append(("get_random_artists", (n,)))
        pool = list(self._artists_by_id.values())
        return random.sample(pool, k=min(n, len(pool)))

    # --- Pending metadata ---------------------------------------------------
    #
    # mark_*/clear_* are idempotent no-ops on repeats, matching the real
    # SqlAlchemyRepository's documented contract: re-marking an already-
    # pending source doesn't reset its created_at (setdefault, not
    # overwrite), and clearing a never-pending source is silently a no-op
    # (dict.pop(..., None)).

    async def mark_artist_metadata_pending(self, artist_id: int, sources: set[str]) -> None:
        self.calls.append(("mark_artist_metadata_pending", (artist_id, frozenset(sources))))
        if not sources:
            return
        pending = self._pending_artist_metadata.setdefault(artist_id, {})
        now = datetime.now(timezone.utc)
        for s in sources:
            pending.setdefault(s, now)

    async def clear_artist_metadata_pending(self, artist_id: int, sources: set[str]) -> None:
        self.calls.append(("clear_artist_metadata_pending", (artist_id, frozenset(sources))))
        if not sources:
            return
        pending = self._pending_artist_metadata.get(artist_id, {})
        for s in sources:
            pending.pop(s, None)

    async def get_stale_pending_artists(self, older_than: datetime) -> list:
        self.calls.append(("get_stale_pending_artists", (older_than,)))
        return [
            self._artists_by_id[artist_id]
            for artist_id, pending in self._pending_artist_metadata.items()
            if artist_id in self._artists_by_id and any(ts < older_than for ts in pending.values())
        ]

    async def mark_song_metadata_pending(self, song_id: int, sources: set[str]) -> None:
        self.calls.append(("mark_song_metadata_pending", (song_id, frozenset(sources))))
        if not sources:
            return
        pending = self._pending_song_metadata.setdefault(song_id, {})
        now = datetime.now(timezone.utc)
        for s in sources:
            pending.setdefault(s, now)

    async def clear_song_metadata_pending(self, song_id: int, sources: set[str]) -> None:
        self.calls.append(("clear_song_metadata_pending", (song_id, frozenset(sources))))
        if not sources:
            return
        pending = self._pending_song_metadata.get(song_id, {})
        for s in sources:
            pending.pop(s, None)

    async def get_stale_pending_songs(self, older_than: datetime) -> list:
        self.calls.append(("get_stale_pending_songs", (older_than,)))
        return [
            self._songs_by_id[song_id]
            for song_id, pending in self._pending_song_metadata.items()
            if song_id in self._songs_by_id and any(ts < older_than for ts in pending.values())
        ]
