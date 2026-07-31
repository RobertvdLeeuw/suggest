"""
Purpose & scope
---------------
Real implementation of SpotifyClientProtocol - the one client kept as a
class, not a module, because Spotify is the one source needing per-instance
state: each user on the machine gets their own SpotifyClient with its own
OAuth token/refresh cycle, unlike MusicBrainz/LastFM (musicbrainz.py,
lastfm.py) which are general-lookup-only and used as module-level
singletons.

Owns: OAuth token refresh, rate-limit/backoff handling, and collecting
paginated results in full. Nothing outside this file should call
time.sleep(), catch a rate-limit error, or loop over `page["next"]` - that
logic lives here once.

spotipy itself is synchronous, so every call runs via asyncio.to_thread.
spotipy also has its own built-in retry (retries=3, status_retries=3,
backoff_factor=0.3 by default) - disabled below (retries=0) so it doesn't
fight with_backoff's retry loop; ours is the only one that runs.

Does NOT know about MusicBrainz, LastFM, the DB, or any ResolvedArtist/Track
type.
"""

import asyncio
import functools

import spotipy
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyOAuth

from .retry import with_backoff, with_retry


def _classify(exc: Exception) -> tuple[str, float] | str:
    """A 429 carries its own Retry-After header - authoritative, no doubling.
    5xx and connection-level failures are retryable but don't know their own
    wait time (wait_s=0 - with_backoff decides). Anything else (401, 404, a
    malformed request) is fatal: retrying it would just fail the same way."""
    if isinstance(exc, SpotifyException):
        if exc.http_status == 429:
            return "retry", float(exc.headers.get("Retry-After", 1))
        if exc.http_status >= 500:
            return "retry"
        return "fatal"
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return "retry"
    return "fatal"


retry = functools.partial(with_retry, _classify)


class SpotifyClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scopes: list[str],
        cache_path: str,
    ):
        self._sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=" ".join(scopes),
                cache_path=cache_path,
            ),
            retries=0,  # with_backoff is the only retry loop that should run
        )

    async def _call(self, fn, *args, **kwargs):
        """Runs a single sync spotipy call in a thread, under with_backoff. Used
        directly (not the @retry decorator) inside paginated methods, so a
        transient failure on page 3 retries just that page - not the whole
        method from page 1."""
        return await with_backoff(lambda: asyncio.to_thread(fn, *args, **kwargs), _classify)

    @retry()
    async def artist(self, artist_id: str) -> dict:
        return await asyncio.to_thread(self._sp.artist, artist_id)

    @retry()
    async def artist_top_tracks(self, artist_id: str) -> dict:
        return await asyncio.to_thread(self._sp.artist_top_tracks, artist_id)

    @retry()
    async def track(self, track_id: str) -> dict:
        return await asyncio.to_thread(self._sp.track, track_id)

    @retry()
    async def search(self, query: str, type: str) -> dict:
        return await asyncio.to_thread(self._sp.search, q=query, type=type)

    @retry()
    async def current_user(self) -> dict:
        return await asyncio.to_thread(self._sp.current_user)

    @retry()
    async def current_playback(self) -> dict | None:
        return await asyncio.to_thread(self._sp.current_playback)

    @retry()
    async def queue(self) -> dict:
        return await asyncio.to_thread(self._sp.queue)

    # --- paginated methods below: retry the per-page call, not the whole method ---

    async def current_user_saved_tracks(self, limit: int = 50) -> dict:
        items = []
        offset = 0
        while True:
            page = await self._call(self._sp.current_user_saved_tracks, limit=limit, offset=offset)
            items.extend(page["items"])
            if page["next"] is None:
                break
            offset += limit
        return {"items": items}

    async def current_user_playlists(self, limit: int = 50) -> dict:
        items = []
        offset = 0
        while True:
            page = await self._call(self._sp.current_user_playlists, limit=limit, offset=offset)
            items.extend(page["items"])
            if page["next"] is None:
                break
            offset += limit
        return {"items": items}

    async def playlist(self, playlist_id: str) -> dict:
        playlist = await self._call(self._sp.playlist, playlist_id)
        items = playlist["tracks"]["items"]
        next_url = playlist["tracks"]["next"]
        offset = len(items)
        while next_url is not None:
            page = await self._call(self._sp.playlist_items, playlist_id, offset=offset)
            items.extend(page["items"])
            next_url = page["next"]
            offset += len(page["items"])
        playlist["tracks"]["items"] = items
        return playlist

    async def album_tracks(self, album_id: str, limit: int = 50) -> dict:
        items = []
        offset = 0
        while True:
            page = await self._call(self._sp.album_tracks, album_id, limit=limit, offset=offset)
            items.extend(page["items"])
            if page["next"] is None:
                break
            offset += limit
        return {"items": items}

    async def artist_albums(self, artist_id: str, limit: int = 50) -> dict:
        items = []
        offset = 0
        while True:
            page = await self._call(self._sp.artist_albums, artist_id, limit=limit, offset=offset)
            items.extend(page["items"])
            if page["next"] is None:
                break
            offset += limit
        return {"items": items}
