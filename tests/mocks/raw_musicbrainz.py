"""
Raw musicbrainzngs-shaped double - what test_clients.py's musicbrainz module
tests call into (search_recordings/get_artist_by_id/get_recording_by_id and
the rate-limit/backoff wrapping around them). Same layer distinction as
raw_spotify.py: this fakes the third-party library, not our Protocol.

clients/musicbrainz.py calls straight through the module-level `mb` name
(`import musicbrainzngs as mb`) rather than through any object we could
inject via a constructor - so tests swap this double in by monkeypatching
that name directly:

    monkeypatch.setattr(collecter.clients.musicbrainz, "mb", RawMusicBrainz())

which is also why this double's public surface mirrors the musicbrainzngs
*module* (set_useragent/auth/search_recordings/get_artist_by_id/
get_recording_by_id) instead of a class instance.

Old apis.py's musicbrainz_fake injected random failures inline
(random.random() < 0.01: raise ResponseError). Don't carry that over -
failure injection should be explicit per-test (via .failures below) so a
failing test is reproducible, not something that occasionally passes by luck.
"""

import time
from typing import Any

import musicbrainzngs as mb

from fixtures.api_responses import MB_ARTIST_BY_ID, MB_RECORDING_BY_ID, MB_SEARCH_RECORDINGS

_DEFAULTS: dict[str, Any] = {
    "search_recordings": MB_SEARCH_RECORDINGS,
    "get_artist_by_id": MB_ARTIST_BY_ID,
    "get_recording_by_id": MB_RECORDING_BY_ID,
}


class _Cause:
    """Minimal stand-in for the underlying urllib HTTPError musicbrainzngs
    wraps into ResponseError.cause - _classify in clients/musicbrainz.py only
    ever reads `.code` off it, so that's all this needs."""

    def __init__(self, code: int):
        self.code = code


def rate_limited_error() -> mb.ResponseError:
    """A 503-wrapped ResponseError - the retryable case _classify checks for."""
    return mb.ResponseError(message="Service Unavailable", cause=_Cause(503))


def fatal_response_error(code: int = 400) -> mb.ResponseError:
    """A plain 4xx-shaped ResponseError - fatal, not retried."""
    return mb.ResponseError(message="Bad Request", cause=_Cause(code))


class RawMusicBrainz:
    """Configurable double for the musicbrainzngs module-level API.

    - self.responses[method_name]: list of return values, popped left-to-right
      across calls. Falls back to _DEFAULTS[method_name] once exhausted (or if
      never configured) - most tests never need to touch this.
    - self.failures[method_name]: an exception to raise on the *next* call to
      that method only (one-shot - popped after raising, so a test can assert
      "fails once, then the retry succeeds" without extra bookkeeping).
    - self.calls: list of (method_name, args, kwargs, timestamp) using
      time.monotonic(), in call order - timestamps are what the "never
      exceeds 1 req/s" rate-limit assertion needs; call count alone can't
      show that.
    """

    def __init__(self):
        self.responses: dict[str, list[Any]] = {}
        self.failures: dict[str, Exception] = {}
        self.calls: list[tuple[str, tuple, dict, float]] = []
        self.configured = False
        self.auth_calls: list[tuple[str, str]] = []

    def _invoke(self, method: str, args: tuple, kwargs: dict) -> Any:
        self.calls.append((method, args, kwargs, time.monotonic()))

        if method in self.failures:
            raise self.failures.pop(method)

        queued = self.responses.get(method)
        if queued:
            return queued.pop(0)

        return _DEFAULTS[method]

    # --- module-level functions clients/musicbrainz.py calls ---------------

    def set_useragent(self, app: str, version: str, contact: str) -> None:
        self.calls.append(("set_useragent", (app, version, contact), {}, time.monotonic()))
        self.configured = True

    def auth(self, username: str, password: str) -> None:
        self.calls.append(("auth", (username, password), {}, time.monotonic()))
        self.auth_calls.append((username, password))

    def search_recordings(self, query: str = None, limit: int = None) -> dict:
        return self._invoke("search_recordings", (), {"query": query, "limit": limit})

    def get_artist_by_id(self, artist_id: str, includes: list[str] | None = None) -> dict:
        return self._invoke("get_artist_by_id", (artist_id,), {"includes": includes})

    def get_recording_by_id(self, recording_id: str, includes: list[str] | None = None) -> dict:
        return self._invoke("get_recording_by_id", (recording_id,), {"includes": includes})

    # --- test-facing helpers -------------------------------------------------

    def call_count(self, method: str) -> int:
        return sum(1 for m, *_ in self.calls if m == method)

    def call_timestamps(self, method: str) -> list[float]:
        return [ts for m, _, _, ts in self.calls if m == method]
