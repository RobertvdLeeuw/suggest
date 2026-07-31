"""
Purpose & scope
---------------
Module-level MusicBrainz client - no class, no instance state. There's only
ever one MB connection per process regardless of how many Spotify users are
active on the machine (MB is used for general track/artist metadata lookup,
never per-user), so a singleton module fits better than a class that would
only ever be instantiated once.

configure() must be called once at startup (main.py) before any other
function here is used - there's no constructor to pass credentials to.

Owns: auth setup, the proactive rate limit (MB asks for max 1 req/s), and
retry/backoff on transient failures. musicbrainzngs itself is synchronous, so
every call is run via asyncio.to_thread to avoid blocking the event loop.

Does NOT know about Spotify, LastFM, the DB, or any ResolvedArtist/Track type.
"""

import asyncio
import functools

import musicbrainzngs as mb

from .retry import with_backoff, with_retry

_MIN_INTERVAL_S = 1.0  # MusicBrainz's documented rate limit: max 1 req/s
_rate_limit = asyncio.Semaphore(1)
_configured = False


def configure(
    app_name: str, app_version: str, contact_email: str, username: str, password: str
) -> None:
    """Call once at process startup. Not safe to call more than once (musicbrainzngs
    itself is process-global state, same as this module)."""
    global _configured
    mb.set_useragent(app_name, app_version, contact_email)
    mb.auth(username, password)
    _configured = True


def _classify(exc: Exception) -> tuple[str, float] | str:
    """MB signals its rate limit as a 503 wrapped in ResponseError.cause; anything
    else wrapping a urllib error (NetworkError, no cause.code at all) is a plain
    connectivity failure - both retryable. A well-formed 4xx (bad query, not found)
    is fatal, not worth retrying."""
    if isinstance(exc, mb.ResponseError) and getattr(exc.cause, "code", None) == 503:
        return "retry"
    if isinstance(exc, mb.NetworkError):
        return "retry"
    return "fatal"


retry = functools.partial(with_retry, _classify)


async def _throttled_to_thread(fn, *args, **kwargs):
    """Runs a sync musicbrainzngs call in a thread, holding the module-wide rate
    limit semaphore for at least _MIN_INTERVAL_S so calls stay paced regardless of
    how many coroutines are calling in concurrently."""
    async with _rate_limit:
        result = await asyncio.to_thread(fn, *args, **kwargs)
        await asyncio.sleep(_MIN_INTERVAL_S)
        return result


@retry()
async def search_recordings(query: str, limit: int | None = None) -> dict:
    assert _configured, "musicbrainz.configure() must be called before use"
    return await _throttled_to_thread(mb.search_recordings, query=query, limit=limit)


@retry()
async def get_artist_by_id(artist_id: str, includes: list[str] | None = None) -> dict:
    assert _configured, "musicbrainz.configure() must be called before use"
    return await _throttled_to_thread(
        mb.get_artist_by_id, artist_id, includes=includes or ["tags", "user-tags"]
    )


@retry()
async def get_recording_by_id(recording_id: str, includes: list[str] | None = None) -> dict:
    assert _configured, "musicbrainz.configure() must be called before use"
    return await _throttled_to_thread(
        mb.get_recording_by_id, recording_id, includes=includes or ["tags", "user-tags"]
    )
