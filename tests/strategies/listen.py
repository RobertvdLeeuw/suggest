"""
Strategies for listen_tracking.py's process_playback_tick - a pure state
machine, so what actually matters here is generating realistic *sequences*
of playback snapshots (play/pause/skip/restart/rewind/track-change
patterns), not just one-off snapshots. A single random snapshot barely
exercises the state machine; a sequence does.

Each element of a sequence is a (snapshot, sleep_time_s) pair, not a bare
snapshot: process_playback_tick takes sleep_time_s per call, and the
skip/fast-forward threshold (sleep_time_s * 5000) is defined relative to it,
so it has to vary per tick alongside the snapshot rather than being fixed
once for a whole sequence - this is also what lets a future test simulate
varying network/poll latency across a single listen.
"""

import hypothesis.strategies as st
from hypothesis import assume
from strategies.apis import spotify_id_strat, spotify_playback_snapshot_strat

Tick = tuple[dict | None, int]

SLEEP_TIME_S = st.integers(min_value=1, max_value=30)


def _track_snapshot(
    track_id: str, duration_ms: int, progress_ms: int, is_playing: bool = True
) -> dict:
    return {
        "is_playing": is_playing,
        "item": {"id": track_id, "type": "track", "duration_ms": duration_ms},
        "progress_ms": progress_ms,
    }


# playback_snapshot_strat: single snapshot dict - reuses/wraps
# strategies.apis.spotify_playback_snapshot_strat rather than redefining the
# shape (see that function's docstring).
playback_snapshot_strat = spotify_playback_snapshot_strat


def next_in_queue_id_strat() -> st.SearchStrategy[str | None]:
    """General-purpose next_in_queue_id: no relation to any particular
    snapshot's track id. Used standalone - e.g. for "a next_in_queue_id
    change alone never emits an event," which only needs *some* value to
    change, not a value that matches anything. For the one pattern where the
    match itself is load-bearing (skipped-via-queue), see
    _queue_skip_conclusion_tick below instead - forcing the match there
    directly is far cheaper than filtering this strategy down to it."""
    return st.one_of(st.none(), spotify_id_strat)


@st.composite
def normal_playthrough_strat(draw) -> list[Tick]:
    """progress_ms increasing monotonically on one track, then a track
    change - the bread-and-butter case exercising ms_played accumulation
    with no rewind/restart/skip branch taken."""
    track_id = draw(spotify_id_strat)
    duration_ms = draw(st.integers(min_value=10_000, max_value=400_000))
    n_ticks = draw(st.integers(min_value=2, max_value=6))

    ticks: list[Tick] = []
    progress = 0
    for _ in range(n_ticks):
        sleep_time_s = draw(SLEEP_TIME_S)
        progress = min(progress + sleep_time_s * 1000, duration_ms - 1)
        ticks.append((_track_snapshot(track_id, duration_ms, progress), sleep_time_s))

    return ticks


@st.composite
def restart_strat(draw) -> list[Tick]:
    """One tick already in progress, then progress_ms drops below
    duration_ms * 0.1 - the restart branch."""
    track_id = draw(spotify_id_strat)
    duration_ms = draw(st.integers(min_value=10_000, max_value=400_000))

    first_sleep = draw(SLEEP_TIME_S)
    mid_progress = draw(st.integers(min_value=int(duration_ms * 0.2), max_value=duration_ms - 1))
    restart_progress = draw(st.integers(min_value=0, max_value=int(duration_ms * 0.1) - 1))
    second_sleep = draw(SLEEP_TIME_S)

    return [
        (_track_snapshot(track_id, duration_ms, mid_progress), first_sleep),
        (_track_snapshot(track_id, duration_ms, restart_progress), second_sleep),
    ]


@st.composite
def rewind_strat(draw) -> list[Tick]:
    """progress_ms drops, but not below the near-start (duration_ms * 0.1)
    threshold - the plain rewind branch, distinct from restart."""
    track_id = draw(spotify_id_strat)
    duration_ms = draw(st.integers(min_value=10_000, max_value=400_000))
    near_start = duration_ms * 0.1

    first_sleep = draw(SLEEP_TIME_S)
    mid_progress = draw(st.integers(min_value=int(near_start) + 2000, max_value=duration_ms - 1))
    rewind_progress = draw(
        st.integers(min_value=int(near_start) + 1, max_value=mid_progress - 1000)
    )
    second_sleep = draw(SLEEP_TIME_S)

    return [
        (_track_snapshot(track_id, duration_ms, mid_progress), first_sleep),
        (_track_snapshot(track_id, duration_ms, rewind_progress), second_sleep),
    ]


@st.composite
def skip_forward_strat(draw) -> list[Tick]:
    """progress_ms jumps forward by more than sleep_time_s * 5000 within the
    same track - the fast-forward/skip-ahead branch. sleep_time_s for the
    jump tick is drawn first since the jump size is defined relative to it."""
    track_id = draw(spotify_id_strat)
    first_sleep = draw(SLEEP_TIME_S)
    duration_ms = draw(st.integers(min_value=200_000, max_value=600_000))
    start_progress = draw(st.integers(min_value=0, max_value=50_000))

    jump_sleep = draw(SLEEP_TIME_S)
    min_jump = jump_sleep * 5000 + 1000
    jump_progress = draw(
        st.integers(
            min_value=start_progress + min_jump,
            max_value=max(start_progress + min_jump + 1, duration_ms - 1),
        )
    )
    jump_progress = min(jump_progress, duration_ms - 1)

    return [
        (_track_snapshot(track_id, duration_ms, start_progress), first_sleep),
        (_track_snapshot(track_id, duration_ms, jump_progress), jump_sleep),
    ]


@st.composite
def pause_resume_strat(draw) -> list[Tick]:
    """is_playing toggles off then on, same track, progress_ms barely
    moving - no track change, so no ListenEvent should fire from this
    alone."""
    track_id = draw(spotify_id_strat)
    duration_ms = draw(st.integers(min_value=10_000, max_value=400_000))
    progress = draw(st.integers(min_value=0, max_value=duration_ms // 2))

    first_sleep = draw(SLEEP_TIME_S)
    second_sleep = draw(SLEEP_TIME_S)

    return [
        (_track_snapshot(track_id, duration_ms, progress, is_playing=False), first_sleep),
        (_track_snapshot(track_id, duration_ms, progress, is_playing=True), second_sleep),
    ]


@st.composite
def queue_skip_strat(draw) -> list[Tick]:
    """A track in progress, then a *different* track whose id matches
    next_in_queue_id - _classify_conclusion's `new_track_id ==
    next_in_queue_id` branch. Forces the id match directly rather than
    drawing two independent ids and filtering for equality (which Hypothesis
    would almost never satisfy by chance).

    duration_ms/sleep bounds are deliberately generous relative to each
    other (large duration, capped sleep) so accumulated ms_played can never
    cross _classify_conclusion's trackdone threshold (checked BEFORE the
    skip check) - otherwise this would sometimes generate a case that looks
    like a skip but actually gets classified as trackdone, silently testing
    the wrong branch."""
    first_track_id = draw(spotify_id_strat)
    next_track_id = draw(spotify_id_strat)
    assume(first_track_id != next_track_id)
    duration_ms = draw(st.integers(min_value=300_000, max_value=600_000))
    progress = draw(st.integers(min_value=0, max_value=int(duration_ms * 0.3)))

    capped_sleep = st.integers(min_value=1, max_value=10)
    first_sleep = draw(capped_sleep)
    second_sleep = draw(capped_sleep)

    return [
        (_track_snapshot(first_track_id, duration_ms, progress), first_sleep),
        (_track_snapshot(next_track_id, duration_ms, 0), second_sleep),
    ]
    # Caller is responsible for passing next_in_queue_id=next_track_id on
    # the SECOND tick only - this strategy generates the snapshots, not the
    # next_in_queue_id argument itself, since that's process_playback_tick's
    # own parameter, not part of a snapshot.


@st.composite
def natural_completion_strat(draw) -> list[Tick]:
    """ms_played reaches >= duration_ms * 0.75 before the track changes -
    the trackdone branch, distinct from a plain track change that hasn't
    played enough to count as "done"."""
    track_id = draw(spotify_id_strat)
    duration_ms = draw(st.integers(min_value=10_000, max_value=100_000))
    next_track_id = draw(spotify_id_strat)
    assume(track_id != next_track_id)

    # One tick whose sleep_time_s alone accounts for >= 75% of duration_ms,
    # then a track change - simplest way to guarantee ms_played crosses the
    # trackdone threshold without a long generated tick sequence.
    sleep_time_s = (duration_ms * 3 // 4) // 1000 + 1
    progress = draw(st.integers(min_value=0, max_value=duration_ms // 4))

    return [
        (_track_snapshot(track_id, duration_ms, progress), sleep_time_s),
        (_track_snapshot(next_track_id, duration_ms, 0), draw(SLEEP_TIME_S)),
    ]


def snapshot_sequence_strat() -> st.SearchStrategy[list[Tick]]:
    """One of the above named patterns, picked at random - covers
    _classify_conclusion's branches collectively rather than hoping pure iid
    randomness stumbles onto each one."""
    return st.one_of(
        normal_playthrough_strat(),
        restart_strat(),
        rewind_strat(),
        skip_forward_strat(),
        pause_resume_strat(),
        queue_skip_strat(),
        natural_completion_strat(),
    )
