"""
Raw pylast-shaped doubles - two separate layers, both needed because
resolution.py talks to pylast objects at two different depths:

  1. RawLastFMNetwork - what test_clients.py's LastFMClient tests call into.
     Only needs get_track() and enable_rate_limit(), since that's all
     clients/lastfm.py's LastFMClient touches on the real
     pylast.LastFMNetwork.

  2. RawTrack / RawArtist - doubles for the pylast.Track/pylast.Artist
     objects resolution.py's LastFM chain calls directly (get_artist,
     get_top_tags, get_similar, get_top_tracks) - NOT LastFMClientProtocol
     methods, they're plain pylast object methods chained off
     LastFMClient.get_track()'s result (or off each other - get_similar
     returns more RawArtists, get_top_tracks returns more RawTracks), so
     they need their own thin doubles rather than reusing
     clients.FakeLastFMClient.

Tag/similarity data here is deliberately synthetic (funky/groovy/krautrock,
not a real API capture) - confirmed with Robert this is fine specifically
for LastFM: pylast already parses the XML into these objects for us, so
there's no raw wire shape left to be faithful to the way there is for
Spotify/MusicBrainz's JSON. Unlike those two, no LASTFM_* fixtures were
added to fixtures/api_responses.py.
"""

import time
from typing import Any

import pylast


class RawTag:
    """Stand-in for a pylast.Tag - only get_name() is ever called on the tag
    half of a TopItem, both by resolution.py's tag flattening
    (_lastfm_top_tags) and by nothing else, so that's all this needs."""

    def __init__(self, name: str):
        self._name = name

    def get_name(self) -> str:
        return self._name


_DEFAULT_TAG_NAMES = ("funky", "groovy", "krautrock")


def default_top_tags() -> list[pylast.TopItem]:
    return [
        pylast.TopItem(item=RawTag(name), weight=100 - i * 10)
        for i, name in enumerate(_DEFAULT_TAG_NAMES)
    ]


def rate_limit_error() -> pylast.WSError:
    """STATUS_RATE_LIMIT_EXCEEDED (29) - the one id _classify_lastfm_failure
    (clients/lastfm.py) and _classify_lastfm_failure (resolution.py) both
    treat as retryable/UNAVAILABLE rather than a confident miss."""
    return pylast.WSError(network=None, status=29, details="rate limit exceeded")


def not_found_error(status: int = 6) -> pylast.WSError:
    """A well-formed error response that isn't the rate limit id - a
    confident miss (fatal in clients/lastfm.py, "no tags"/None rather than
    UNAVAILABLE in resolution.py's _classify_lastfm_failure)."""
    return pylast.WSError(network=None, status=status, details="invalid parameters")


class RawTrack:
    """Double for pylast.Track - what LastFMClient.get_track() returns, and
    what resolution.py's _sp_artist_to_lastfm/_lastfm_top_tags chain off of
    directly (.get_artist(), .get_top_tags()).

    self.failures[method_name]: exception to raise on the next call to that
    method only (one-shot), same convention as the raw_spotify.py/
    raw_musicbrainz.py doubles.
    """

    def __init__(
        self,
        title: str,
        artist: "RawArtist | None" = None,
        top_tags: list[pylast.TopItem] | None = None,
    ):
        self._title = title
        self._artist = artist if artist is not None else RawArtist(name="Default Artist")
        self._top_tags = top_tags if top_tags is not None else default_top_tags()
        self.failures: dict[str, Exception] = {}
        self.calls: list[str] = []

    def _resolve(self, method: str, value):
        self.calls.append(method)
        if method in self.failures:
            raise self.failures.pop(method)
        return value

    def get_title(self) -> str:
        return self._resolve("get_title", self._title)

    def get_artist(self) -> "RawArtist":
        return self._resolve("get_artist", self._artist)

    def get_top_tags(self) -> list[pylast.TopItem]:
        return self._resolve("get_top_tags", self._top_tags)


class RawArtist:
    """Double for pylast.Artist - what RawTrack.get_artist() and
    RawArtist.get_similar()/.get_top_tracks() return, and what
    resolution.py's get_similar_artists/_lastfm_to_sp chain off of directly
    (.get_name(), .get_top_tags(), .get_similar(), .get_top_tracks()).

    get_similar's default is empty (not every artist has similar-artist data
    worth asserting on by default); a test exercising get_similar_artists
    needs to configure it explicitly via the constructor or by mutating
    ._similar - there's no sensible "default similar artist" to invent.
    """

    def __init__(
        self,
        name: str,
        top_tags: list[pylast.TopItem] | None = None,
        similar: list[pylast.SimilarItem] | None = None,
        top_tracks: list[pylast.TopItem] | None = None,
    ):
        self._name = name
        self._top_tags = top_tags if top_tags is not None else default_top_tags()
        self._similar = similar if similar is not None else []
        self._top_tracks = top_tracks if top_tracks is not None else []
        self.failures: dict[str, Exception] = {}
        self.calls: list[str] = []

    def _resolve(self, method: str, value):
        self.calls.append(method)
        if method in self.failures:
            raise self.failures.pop(method)
        return value

    def get_name(self) -> str:
        return self._resolve("get_name", self._name)

    def get_top_tags(self) -> list[pylast.TopItem]:
        return self._resolve("get_top_tags", self._top_tags)

    def get_similar(self, limit: int = 3) -> list[pylast.SimilarItem]:
        return self._resolve("get_similar", self._similar[:limit])

    def get_top_tracks(self) -> list[pylast.TopItem]:
        return self._resolve("get_top_tracks", self._top_tracks)


class RawLastFMNetwork:
    """Double for pylast.LastFMNetwork - only get_track() and
    enable_rate_limit(), matching clients/lastfm.py's LastFMClient's actual
    surface.

    - self.responses["get_track"]: list of RawTrack instances, popped
      left-to-right across calls (for tests that want get_track to return
      different tracks on successive calls). Falls back to a fresh default
      RawTrack(title=title) if never configured/exhausted.
    - self.failures["get_track"]: an exception to raise on the *next* call
      only (one-shot).
    - self.calls: list of (method, args, kwargs, timestamp) in call order.
    """

    def __init__(self):
        self.responses: dict[str, list[Any]] = {}
        self.failures: dict[str, Exception] = {}
        self.calls: list[tuple[str, tuple, dict, float]] = []
        self.rate_limit_enabled = False

    def enable_rate_limit(self) -> None:
        self.rate_limit_enabled = True

    def get_track(self, artist: str, title: str) -> RawTrack:
        self.calls.append(("get_track", (), {"artist": artist, "title": title}, time.monotonic()))

        if "get_track" in self.failures:
            raise self.failures.pop("get_track")

        queued = self.responses.get("get_track")
        if queued:
            return queued.pop(0)

        return RawTrack(title=title, artist=RawArtist(name=artist))

    # --- test-facing helpers -------------------------------------------------

    def call_count(self, method: str = "get_track") -> int:
        return sum(1 for m, *_ in self.calls if m == method)
