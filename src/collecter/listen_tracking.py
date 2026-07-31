"""
Purpose & scope
---------------
Pure state machine for turning a stream of Spotify playback snapshots (from
polling sp.current_playback()) into listen events and chunks - rewind/
fast-forward/restart/skip detection, ms_played accumulation, reason_start/
reason_end classification. This is the logic that used to live inline inside
add_recent_listen_loop in metadata.py.

Rules for this file:
  - No I/O. No sqlalchemy, no client Protocols, no asyncio.sleep. Takes a
    state + a snapshot in, returns a new state + optional event out.
  - services.py owns the actual polling loop (asyncio.sleep, calling
    spotify.current_playback(), persisting events via repository.py) and
    calls into this module once per tick. That split is what lets the chunk
    math and reason-transition rules be tested with synthetic snapshot
    sequences and zero network/DB.
"""

from dataclasses import dataclass, field


@dataclass
class ListenChunk:
    from_ms: int
    to_ms: int


@dataclass
class ListenEvent:
    spotify_id: str
    ms_played: int
    reason_start: str
    reason_end: str
    chunks: list[ListenChunk]


@dataclass
class TrackingState:
    current_listen: dict | None = None
    reason_start: str = "unknown"
    next_in_queue_id: str | None = None
    ms_played: int = 0
    chunks: list[ListenChunk] = field(default_factory=list)
    latest_chunk_start: int = 0


def process_playback_tick(
    state: TrackingState,
    new_snapshot: dict | None,
    next_in_queue_id: str | None,
    sleep_time_s: int,
) -> tuple[TrackingState, ListenEvent | None]:
    """Advances the state machine by one poll tick. Returns the new state, and a
    ListenEvent if a listen just concluded (song changed, restarted, etc.) and should
    be persisted by the caller - None otherwise."""
    ...
