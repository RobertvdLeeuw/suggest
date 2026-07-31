"""
Purpose & scope
---------------
Real implementation of LastFMClientProtocol. Owns auth and rate limiting for
LastFM calls.

Does NOT know about Spotify, MusicBrainz, the DB, or any ResolvedArtist/Track
type. Pure wrapper around pylast.
"""

import asyncio
import functools

import pylast

from .retry import with_retry


def classify_lastfm(exc: Exception) -> tuple[str, float] | str:
    """LastFM signals its rate limit as WSError id 29 (STATUS_RATE_LIMIT_EXCEEDED) -
    retryable. Anything else wrapped in a WSError (bad params, not found, auth
    failure) is fatal: retrying it would just fail the same way."""
    if isinstance(exc, pylast.WSError):
        return "retry" if exc.get_id() == 29 else "fatal"
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return "retry"
    return "fatal"


retry = functools.partial(with_retry, classify_lastfm)


class LastFMClient:
    def __init__(self, api_key: str, api_secret: str, username: str, password_hash: str):
        self._lastfm = pylast.LastFMNetwork(
            api_key=api_key,
            api_secret=api_secret,
            username=username,
            password_hash=password_hash,
        )
        self._lastfm.enable_rate_limit()

    @retry()
    async def get_track(self, artist: str, title: str):
        return await asyncio.to_thread(self._lastfm.get_track, artist, title)
