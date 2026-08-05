"""
Raw spotipy.Spotify-shaped double - what test_clients.py's SpotifyClient
tests call into. NOT the same layer as clients.FakeSpotifyClient: this one
stands in for the third-party library itself (spotipy), so
clients/spotify.py's real pagination/backoff/_classify code runs against it
unmodified. clients.FakeSpotifyClient (in this same dir) stands in for OUR
SpotifyClientProtocol instead, one layer up - resolution.py/services.py
tests use that one, never this one.

Unlike raw_musicbrainz.py, SpotifyClient takes its spotipy.Spotify instance
via constructor (`self._sp = spotipy.Spotify(...)`), so this double gets
injected by constructing SpotifyClient normally and then swapping the
instance attribute: `client._sp = RawSpotify()` - no monkeypatching of a
module-level name needed here.

Default responses come from fixtures/api_responses.py's SPOTIFY_* constants -
real captured shapes, not invented ones. Per-test configuration (raising,
paginating, going stale) overrides those defaults per-method via
.responses/.failures, never by hand-editing the fixtures.
"""

import time
from typing import Any

from spotipy.exceptions import SpotifyException

from fixtures.api_responses import (
    SPOTIFY_ALBUM_TRACKS,
    SPOTIFY_ARTIST,
    SPOTIFY_ARTIST_ALBUMS,
    SPOTIFY_ARTIST_TOP_TRACKS,
    SPOTIFY_CURRENT_PLAYBACK,
    SPOTIFY_CURRENT_USER,
    SPOTIFY_CURRENT_USER_PLAYLISTS,
    SPOTIFY_PLAYLIST,
    SPOTIFY_PLAYLIST_ITEMS,
    SPOTIFY_QUEUE,
    SPOTIFY_SAVED_TRACKS,
    SPOTIFY_SEARCH,
    SPOTIFY_TRACK,
)

def _terminal_page(page: dict) -> dict:
    """Golden fixture pages were captured as real first-page API responses, so
    each one has a genuine 'next' URL baked in (the account they came from
    had more than one page of data). Used verbatim as a *default* return,
    that would make the real client's paginate-until-next-is-None loop spin
    forever calling this double back-to-back. Defaults are presented as a
    single terminal page instead (next forced to None) so a test that
    doesn't care about pagination can call a paginated method with zero
    config and get one sane page back. Tests that want to exercise real
    pagination configure .responses[method] with their own explicit
    multi-page sequence - either hand-built, or via paginate() below.
    """
    return {**page, "next": None}


def _terminal_playlist(playlist: dict) -> dict:
    """Same idea as _terminal_page, one level down - playlist()'s "next" lives
    at playlist["tracks"]["next"], not at the top level."""
    return {**playlist, "tracks": _terminal_page(playlist["tracks"])}


def paginate(items: list, page_size: int) -> list[dict]:
    """Deterministically chunk a flat item list into a chain of page dicts
    ({"items": [...], "next": <index into the returned list, or None>}),
    each page's "next" pointing at the next chunk and the last page's "next"
    forced to None - matches the shape SpotifyClient's pagination loop walks.

    Deliberately has no randomness of its own: *how many* items/what page
    size to use is a decision for strategies/apis.py's Hypothesis strategies
    (or a test writing it by hand) to make and shrink on, not something this
    mock should be generating unseeded. This only handles the mechanical
    "turn N items into correctly-chained pages" part.

    The returned list is meant to be assigned directly to
    RawSpotify().responses[method_name] - each dict popped off in order IS
    the page returned to that call, so "next" here doesn't need to be a real
    URL, just non-None on every page but the last (the real client only ever
    checks truthiness).
    """
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    pages = [items[i : i + page_size] for i in range(0, len(items), page_size)] or [[]]
    return [
        {"items": page, "next": "next-page" if i < len(pages) - 1 else None}
        for i, page in enumerate(pages)
    ]


_DEFAULTS: dict[str, Any] = {
    "artist": SPOTIFY_ARTIST,
    "artist_top_tracks": SPOTIFY_ARTIST_TOP_TRACKS,
    "track": SPOTIFY_TRACK,
    "search": SPOTIFY_SEARCH,
    "current_user": SPOTIFY_CURRENT_USER,
    "current_playback": SPOTIFY_CURRENT_PLAYBACK,
    "queue": SPOTIFY_QUEUE,
    "current_user_saved_tracks": _terminal_page(SPOTIFY_SAVED_TRACKS),
    "current_user_playlists": _terminal_page(SPOTIFY_CURRENT_USER_PLAYLISTS),
    "playlist": _terminal_playlist(SPOTIFY_PLAYLIST),
    "playlist_items": _terminal_page(SPOTIFY_PLAYLIST_ITEMS),
    "album_tracks": _terminal_page(SPOTIFY_ALBUM_TRACKS),
    "artist_albums": _terminal_page(SPOTIFY_ARTIST_ALBUMS),
}


def rate_limited_error(retry_after: float = 1.0) -> SpotifyException:
    """A 429 with a Retry-After header - _classify reads the header directly
    (no doubling), so a test asserting an exact honored wait time needs this,
    not a generic 5xx."""
    return SpotifyException(
        http_status=429, code=-1, msg="rate limited", headers={"Retry-After": str(retry_after)}
    )


def server_error(status: int = 503) -> SpotifyException:
    """5xx - retryable, but (unlike 429) doesn't carry its own wait time."""
    return SpotifyException(http_status=status, code=-1, msg="server error")


def fatal_error(status: int = 401) -> SpotifyException:
    """A non-429, non-5xx failure (bad auth, not found, malformed request) -
    fatal, not retried."""
    return SpotifyException(http_status=status, code=-1, msg="fatal error")


class RawSpotify:
    """Configurable double for the raw spotipy.Spotify client
    clients/spotify.py's SpotifyClient wraps.

    - self.responses[method_name]: list of return values, popped
      left-to-right across calls. For paginated methods this is how a test
      drives real multi-page walks (queue one dict per page, each with the
      "next" value the real client should see next; last page's "next" must
      be None or the walk never terminates). Falls back to
      _DEFAULTS[method_name] once exhausted (or if never configured).
    - self.failures[method_name]: an exception to raise on the *next* call to
      that method only (one-shot - popped after raising).
    - self.calls: list of (method_name, args, kwargs, timestamp) in call
      order, using time.monotonic() - needed for retry/backoff timing
      assertions, not just call counts.
    """

    def __init__(self):
        self.responses: dict[str, list[Any]] = {}
        self.failures: dict[str, Exception] = {}
        self.calls: list[tuple[str, tuple, dict, float]] = []

    def _invoke(self, method: str, args: tuple, kwargs: dict) -> Any:
        self.calls.append((method, args, kwargs, time.monotonic()))

        if method in self.failures:
            raise self.failures.pop(method)

        queued = self.responses.get(method)
        if queued:
            return queued.pop(0)

        return _DEFAULTS[method]

    # --- single-call methods -------------------------------------------------

    def artist(self, artist_id: str) -> dict:
        return self._invoke("artist", (artist_id,), {})

    def artist_top_tracks(self, artist_id: str) -> dict:
        return self._invoke("artist_top_tracks", (artist_id,), {})

    def track(self, track_id: str) -> dict:
        return self._invoke("track", (track_id,), {})

    def search(self, q: str, type: str) -> dict:
        return self._invoke("search", (), {"q": q, "type": type})

    def current_user(self) -> dict:
        return self._invoke("current_user", (), {})

    def current_playback(self) -> dict | None:
        return self._invoke("current_playback", (), {})

    def queue(self) -> dict:
        return self._invoke("queue", (), {})

    # --- paginated methods -----------------------------------------------

    def current_user_saved_tracks(self, limit: int = 50, offset: int = 0) -> dict:
        return self._invoke("current_user_saved_tracks", (), {"limit": limit, "offset": offset})

    def current_user_playlists(self, limit: int = 50, offset: int = 0) -> dict:
        return self._invoke("current_user_playlists", (), {"limit": limit, "offset": offset})

    def playlist(self, playlist_id: str) -> dict:
        return self._invoke("playlist", (playlist_id,), {})

    def playlist_items(self, playlist_id: str, offset: int = 0) -> dict:
        return self._invoke("playlist_items", (playlist_id,), {"offset": offset})

    def album_tracks(self, album_id: str, limit: int = 50, offset: int = 0) -> dict:
        return self._invoke("album_tracks", (album_id,), {"limit": limit, "offset": offset})

    def artist_albums(self, artist_id: str, limit: int = 50, offset: int = 0) -> dict:
        return self._invoke("artist_albums", (artist_id,), {"limit": limit, "offset": offset})

    # --- test-facing helpers -------------------------------------------------

    def call_count(self, method: str) -> int:
        return sum(1 for m, *_ in self.calls if m == method)

    def call_timestamps(self, method: str) -> list[float]:
        return [ts for m, _, _, ts in self.calls if m == method]
