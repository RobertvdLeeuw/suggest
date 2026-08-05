"""
Purpose & scope
---------------
Orchestration only, peer to services.py - the actual download loop and
downloads-folder cleanup. Glues download_tracking.py's pure plan_downloads/
plan_cleanup/match_on_disk to the real filesystem, repository.py, and a
DownloaderClientProtocol. This is what main.py runs alongside
services.run_recent_listen_loop and the embedders/ worker processes.

Replaces old/downloader.py's start_download_loop/_download/clean_downloads.
Two behavioral changes from that version:
  - A spotify_id wanted by more than one embedding queue is downloaded once
    and fanned out to every queue that wanted it (see targets_for()), not
    downloaded once per queue.
  - Cleanup runs as a periodic step inside this same loop rather than a
    separate BackgroundScheduler thread. old/downloader.py's clean_downloads
    read the module-global CURRENTLY_DOWNLOADING from a different thread
    than the one mutating it; running both on the same asyncio task means
    DownloadState is only ever touched from one place, no cross-thread
    synchronization needed.

Rules for this file:
  - Only file doing filesystem I/O for downloads (os.listdir/os.remove) -
    download_tracking.py never touches the filesystem itself.
  - DownloadState is a local variable owned by download_loop, not a module
    global - no cross-test/cross-run leakage, same rule services.py's
    docstring gives repository.py's transactional units.
"""

import asyncio
import logging
import os
import time
import traceback
from pathlib import Path

from .clients.downloader import DOWNLOAD_ERRORS, DownloaderClientProtocol
from .download_tracking import (
    DownloadState,
    match_on_disk,
    plan_cleanup,
    plan_downloads,
    resolve_downloads,
    targets_for,
)
from .embedders.song_queue import QUEUE_MAX_LEN, SongQueue
from .repository import Repository

LOGGER = logging.getLogger(__name__)


async def _download_one(
    spotify_id: str,
    downloader: DownloaderClientProtocol,
    repo: Repository,
    queues_by_name: dict[str, SongQueue],
    target_names: list[str],
) -> None:
    """Searches, downloads, renames, and fans the result out to every queue
    in target_names. A handled download failure (DOWNLOAD_ERRORS) dequeues
    the id from each target queue's DB table and returns - matches
    old/downloader.py's "remove from queue on failure" behavior, just
    through repo.dequeue_track instead of a raw sqlalchemy delete() living
    in this file. Anything else propagates - download_loop's gather(...,
    return_exceptions=True) is what catches and logs it, not this
    function."""
    try:
        candidate = await downloader.search(spotify_id)
        if candidate is None:
            LOGGER.warning(f"No song found for id: {spotify_id}")
            return

        LOGGER.info(f"Downloading '{spotify_id}': {candidate.name} by {candidate.artist}")
        file_path = await downloader.download(candidate)
    except DOWNLOAD_ERRORS as e:
        LOGGER.warning(f"{spotify_id} download failed: {e}")
        for name in target_names:
            await repo.dequeue_track(queues_by_name[name].q_type, spotify_id)
        return

    new_path = file_path.parent / f"{spotify_id}_{file_path.name}"
    if file_path != new_path and not new_path.exists():
        file_path.rename(new_path)
        file_path = new_path

    for name in target_names:
        queues_by_name[name].put((str(file_path), spotify_id))

    LOGGER.info(f"Downloaded and queued '{spotify_id}' for: {', '.join(target_names)}")


async def download_loop(
    song_queues: list[SongQueue],
    repo: Repository,
    downloader: DownloaderClientProtocol,
    download_dir: str,
    sleep_time_s: int = 1,
    cleanup_interval_s: int = 60,
) -> None:
    """The actual loop: sleeps, gathers one tick's inputs (a DB query per
    queue, a peek_all() per queue, one listdir()), asks plan_downloads what
    to do, applies it, and periodically runs cleanup - same tick shape as
    services.run_recent_listen_loop. Runs forever; intended to be one of the
    tasks in main.py's asyncio.gather."""
    queues_by_name = {q.name: q for q in song_queues}
    state = DownloadState()
    last_cleanup = time.monotonic()

    LOGGER.info(f"Download loop started ({download_dir}).")

    while True:
        await asyncio.sleep(sleep_time_s)
        try:
            wanted_by_queue = {
                q.name: (
                    await repo.get_queued_track_ids(q.q_type, limit=QUEUE_MAX_LEN - len(q))
                    if len(q) < QUEUE_MAX_LEN
                    else []
                )
                for q in song_queues
            }
            already_local = {
                q.name: {spotify_id for _, spotify_id in q.peek_all()} for q in song_queues
            }

            all_wanted = {sid for ids in wanted_by_queue.values() for sid in ids}
            filenames = await asyncio.to_thread(os.listdir, download_dir)
            on_disk = match_on_disk(all_wanted, filenames, download_dir)

            state, to_download, queue_puts = plan_downloads(
                state, wanted_by_queue, already_local, on_disk
            )

            for queue_name, spotify_id, filepath in queue_puts:
                queues_by_name[queue_name].put((filepath, spotify_id))

            if to_download:
                results = await asyncio.gather(
                    *[
                        _download_one(
                            spotify_id,
                            downloader,
                            repo,
                            queues_by_name,
                            targets_for(wanted_by_queue, spotify_id),
                        )
                        for spotify_id in to_download
                    ],
                    return_exceptions=True,
                )
                for spotify_id, result in zip(to_download, results):
                    if isinstance(result, Exception):
                        LOGGER.error(f"Unexpected error downloading {spotify_id}: {result!r}")

                state = resolve_downloads(state, to_download)

            if time.monotonic() - last_cleanup >= cleanup_interval_s:
                await _clean_downloads(song_queues, state, download_dir)
                last_cleanup = time.monotonic()

        except KeyboardInterrupt:
            raise
        except Exception:
            LOGGER.error(f"Error in download loop: {traceback.format_exc()}")


async def _clean_downloads(
    song_queues: list[SongQueue], state: DownloadState, download_dir: str
) -> int:
    """Deletes any downloaded file not protected by an in-flight download or
    a still-pending item in some queue's local SongQueue. Called from inside
    download_loop on a timer (see cleanup_interval_s) rather than a separate
    scheduled job - see module docstring for why."""
    protected = set(state.in_flight)
    for q in song_queues:
        with q.lock:
            protected.update(spotify_id for _, spotify_id in q.queue)

    files = await asyncio.to_thread(os.listdir, download_dir)
    to_delete = plan_cleanup(protected, files)

    for fname in to_delete:
        await asyncio.to_thread(os.remove, os.path.join(download_dir, fname))

    LOGGER.info(f"Cleaned downloads folder: removed {len(to_delete)} of {len(files)} files.")
    return len(to_delete)
