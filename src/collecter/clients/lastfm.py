"""
Purpose & scope
---------------
Real implementation of LastFMClientProtocol. Owns auth and rate limiting for
LastFM calls.

Does NOT know about Spotify, MusicBrainz, the DB, or any ResolvedArtist/Track
type. Pure wrapper around pylast.
"""

from retry import with_retry

from . import LastFMClientProtocol


class LastFMClient:
    def __init__(self, api_key: str, api_secret: str, username: str, password_hash: str): ...

    @with_retry(classify_lastfm)
    def get_track(self, artist: str, title: str): ...


def classify_lastfm(exc: Exception) -> tuple[str, float] | str:
    if isinstance(exc, pylast.WSError):
        return "retry" if exc.get_id() == 29 else "fatal"  # 29 = STATUS_RATE_LIMIT_EXCEEDED
    return "fatal"
