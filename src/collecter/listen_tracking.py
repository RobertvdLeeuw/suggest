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

from dataclasses import dataclass, field, replace


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


def _close_chunk(chunks: list[ListenChunk], from_ms: int, to_ms: int) -> list[ListenChunk]:
    """Appends a chunk covering [from_ms, to_ms) - unless it'd be zero-width."""
    return [*chunks, ListenChunk(from_ms, to_ms)] if to_ms > from_ms else chunks


def _classify_conclusion(
    ms_played: int,
    duration_ms: int,
    new_track_id: str,
    next_in_queue_id: str | None,
    was_playing: bool,
) -> tuple[str, str]:
    """Decides (reason_end, new_reason_start) for a listen that just concluded
    because the track changed."""
    if ms_played >= duration_ms * 0.75:
        return "trackdone", "trackdone"
    if new_track_id == next_in_queue_id:
        return "skipped", "skipped"
    if not was_playing:
        return "paused", "selected"
    return "unknown", "unknown"


def process_playback_tick(
    state: TrackingState,
    new_snapshot: dict | None,
    next_in_queue_id: str | None,
    sleep_time_s: int,
) -> tuple[TrackingState, ListenEvent | None]:
    """Advances the state machine by one poll tick. Returns the new state, and a
    ListenEvent if a listen just concluded (song changed, restarted, etc.) and should
    be persisted by the caller - None otherwise."""
    if (
        not new_snapshot
        or not new_snapshot.get("item")
        or not new_snapshot["is_playing"]
        or new_snapshot["item"]["type"] != "track"
    ):
        return replace(state, next_in_queue_id=next_in_queue_id), None

    ms_played = state.ms_played + sleep_time_s * 1000

    if state.current_listen is None:
        return (
            replace(
                state,
                current_listen=new_snapshot,
                next_in_queue_id=next_in_queue_id,
                ms_played=ms_played,
            ),
            None,
        )

    current_listen = state.current_listen
    chunks = state.chunks
    latest_chunk_start = state.latest_chunk_start
    reason_start = state.reason_start

    if current_listen["item"]["id"] == new_snapshot["item"]["id"]:
        near_start = current_listen["item"]["duration_ms"] * 0.1
        event = None

        if new_snapshot["progress_ms"] < current_listen["progress_ms"]:
            chunks = _close_chunk(chunks, latest_chunk_start, current_listen["progress_ms"])
            latest_chunk_start = new_snapshot["progress_ms"]

            if new_snapshot["progress_ms"] < near_start:  # Restart.
                event = ListenEvent(
                    spotify_id=current_listen["item"]["id"],
                    ms_played=ms_played,
                    reason_start=reason_start,
                    reason_end="restarted",
                    chunks=chunks,
                )
                reason_start = "restarted"
                chunks = []
                latest_chunk_start = 0
        elif new_snapshot["progress_ms"] - current_listen["progress_ms"] > sleep_time_s * 5000:
            chunks = _close_chunk(chunks, latest_chunk_start, current_listen["progress_ms"])
            latest_chunk_start = new_snapshot["progress_ms"]

        return (
            TrackingState(
                current_listen=new_snapshot,
                reason_start=reason_start,
                next_in_queue_id=next_in_queue_id,
                ms_played=ms_played,
                chunks=chunks,
                latest_chunk_start=latest_chunk_start,
            ),
            event,
        )

    # New song - the previous one just concluded.
    reason_end, new_reason_start = _classify_conclusion(
        ms_played,
        current_listen["item"]["duration_ms"],
        new_snapshot["item"]["id"],
        next_in_queue_id,
        current_listen["is_playing"],
    )
    chunks = _close_chunk(chunks, latest_chunk_start, current_listen["progress_ms"])

    event = ListenEvent(
        spotify_id=current_listen["item"]["id"],
        ms_played=ms_played,
        reason_start=reason_start,
        reason_end=reason_end,
        chunks=chunks,
    )

    return (
        TrackingState(
            current_listen=new_snapshot,
            reason_start=new_reason_start,
            next_in_queue_id=next_in_queue_id,
            ms_played=0,
            chunks=[],
            latest_chunk_start=0,
        ),
        event,
    )
