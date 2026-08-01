"""
Purpose & scope
---------------
Pure decision logic for the download loop and downloads-folder cleanup -
given what each embedding queue's DB table wants, what's already sitting in
each queue's in-memory SongQueue, and what's already on disk, decide which
spotify_ids need an actual download, which queues should receive which
files, and which files clean_downloads should delete.

This is the part of old/downloader.py's start_download_loop/clean_downloads
that was hardest to test: nested loops over os.listdir() results mixed with
mp.Lock-guarded queue state and a module-global CURRENTLY_DOWNLOADING set.
Pulling the decision-making out into plain functions over plain data (no
filesystem, no DB, no locks, no globals) makes every branch testable with a
handful of dicts/sets/lists - same split as listen_tracking.py's
process_playback_tick vs. services.run_recent_listen_loop.

Rules for this file:
  - No I/O. No os.listdir/os.remove, no sqlalchemy, no SongQueue/mp.Lock, no
    asyncio. download.py gathers the inputs (one listdir, one DB query per
    queue, one peek_all() per queue) and applies the outputs (queue.put,
    os.remove).
  - Downloads are deduplicated across queues here: a spotify_id wanted by
    more than one queue is downloaded once; download.py fans the result out
    to every queue that wanted it via targets_for().
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class DownloadState:
    """in_flight is shared across every queue, not per-queue - a track being
    downloaded to satisfy JukeMIR must not also be downloaded to satisfy
    Auditus. Owned by whoever runs download_loop (see download.py) as a
    plain local variable - never a module global, so nothing leaks between
    runs or between tests."""

    in_flight: frozenset[str] = frozenset()


def match_on_disk(wanted_ids: set[str], filenames: list[str], download_dir: str) -> dict[str, str]:
    """Maps each wanted id to a filepath if some file on disk contains it in
    the name (download.py names files `{spotify_id}_{original_name}`, same
    as old/downloader.py). Plain string matching over an already-fetched
    filename list - no filesystem access here, download.py does the one
    os.listdir() per tick and hands the result in."""
    result: dict[str, str] = {}
    for spotify_id in wanted_ids:
        for fname in filenames:
            if spotify_id in fname:
                result[spotify_id] = f"{download_dir}/{fname}"
                break
    return result


def plan_downloads(
    state: DownloadState,
    wanted_by_queue: dict[str, list[str]],
    already_local: dict[str, set[str]],
    on_disk: dict[str, str],
) -> tuple[DownloadState, list[str], list[tuple[str, str, str]]]:
    """
    wanted_by_queue: queue name -> spotify_ids that queue's DB table wants,
        oldest first (already limited to queue capacity by the caller).
    already_local: queue name -> spotify_ids already sitting in that queue's
        in-memory SongQueue (from peek_all()).
    on_disk: spotify_id -> filepath, from match_on_disk() against one
        listdir() this tick.

    Returns (new_state, ids_to_download, queue_puts):
      new_state: in_flight extended with ids_to_download.
      ids_to_download: deduped ids that need an actual download this tick -
        not already_local anywhere that wants them, not on_disk, not
        already in_flight from a previous tick.
      queue_puts: (queue_name, spotify_id, filepath) to insert into a
        queue's local SongQueue immediately, no download needed - found on
        disk already. Ids that are in_flight are NOT included here; they
        resolve once the in-progress download completes, via
        download.py + targets_for(), not through this tick's queue_puts.
    """
    queue_puts: list[tuple[str, str, str]] = []
    to_download: list[str] = []
    seen: set[str] = set()

    for queue_name, wanted_ids in wanted_by_queue.items():
        local = already_local.get(queue_name, set())

        for spotify_id in wanted_ids:
            if spotify_id in local:
                continue

            if spotify_id in on_disk:
                queue_puts.append((queue_name, spotify_id, on_disk[spotify_id]))
                continue

            if spotify_id in state.in_flight or spotify_id in seen:
                continue

            to_download.append(spotify_id)
            seen.add(spotify_id)

    new_state = replace(state, in_flight=state.in_flight | seen)
    return new_state, to_download, queue_puts


def resolve_downloads(state: DownloadState, spotify_ids: list[str]) -> DownloadState:
    """Call once a batch of dispatched downloads has finished (success,
    handled failure, or unexpected exception - download.py awaits all three
    outcomes via asyncio.gather(..., return_exceptions=True) before calling
    this, so nothing here is still in progress). Clears them from in_flight:
    a failed one becomes retryable next tick, a successful one is from then
    on tracked via already_local instead."""
    return replace(state, in_flight=state.in_flight - set(spotify_ids))


def targets_for(wanted_by_queue: dict[str, list[str]], spotify_id: str) -> list[str]:
    """Which queue names wanted this id - used by download.py to fan a
    freshly-completed download out to every queue that asked for it, not
    just whichever queue's tick happened to trigger the download."""
    return [name for name, ids in wanted_by_queue.items() if spotify_id in ids]


def plan_cleanup(protected_ids: set[str], files_on_disk: list[str]) -> list[str]:
    """Which files to delete: any file whose name doesn't contain a
    protected spotify_id. Same rule as old/downloader.py's clean_downloads -
    a file stays as long as ANY queue (or an in-flight download) still wants
    it, so JukeMIR finishing first never deletes a file Auditus still needs."""
    return [f for f in files_on_disk if not any(sid in f for sid in protected_ids)]
