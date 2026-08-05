"""
Fake implementations of SpotifyClientProtocol/MusicBrainzClientProtocol/
LastFMClientProtocol/DownloaderClientProtocol (see collecter.clients and
collecter.clients.downloader) - the layer resolution.py/services.py/
download.py code against. Configurable per-test, not random - hypothesis
strategies (strategies/resolution.py, strategies/download.py) drive the
randomness, these just return what they're told to.

Distinct from mocks/raw_*.py: those fake the third-party libraries
(spotipy/musicbrainzngs/pylast) one layer further down, for testing OUR
wrapper code in clients/*.py. Nothing here should import spotipy,
musicbrainzngs, or pylast.
"""

from pathlib import Path
from typing import Any

from src.collecter.clients import LastFMClientProtocol, MusicBrainzClientProtocol, SpotifyClientProtocol
from src.collecter.clients.downloader import DownloadCandidate, DownloaderClientProtocol


class UnconfiguredCall(AssertionError):
    """Raised when a Fake*Client method is called with no response/failure
    configured for it. Deliberately loud instead of silently returning None -
    this layer exists specifically for per-test configuration, so a test
    that forgot to configure a call it depends on should fail right at that
    call, not several lines later on a confusing None/KeyError."""


class _AsyncFake:
    """Shared plumbing for the three simple Protocol fakes below. Each public
    method just calls self._invoke(method_name, args, kwargs) and returns
    the result - no per-method logic of its own.

    - self.responses[method_name]: list of return values, popped
      left-to-right across calls to that method.
    - self.by_args[method_name]: dict[args_tuple, value] - for tests that
      care about *which* id was looked up (e.g. a push_artist test calling
      spotify.artist() for two different ids in one test and needing two
      different answers) rather than call order. Checked before .responses,
      since it's the more specific configuration.
    - self.failures[method_name]: exception to raise on the *next* call to
      that method only (one-shot - popped after raising).
    - self.calls: list[(method_name, args, kwargs)], in call order - what
      services.py's ordering/short-circuit assertions check.

    Nothing is configured by default - an unconfigured call raises
    UnconfiguredCall rather than silently returning something plausible.
    """

    def __init__(self):
        self.responses: dict[str, list[Any]] = {}
        self.by_args: dict[str, dict[tuple, Any]] = {}
        self.failures: dict[str, Exception] = {}
        self.calls: list[tuple[str, tuple, dict]] = []

    def _invoke(self, method: str, args: tuple, kwargs: dict) -> Any:
        self.calls.append((method, args, kwargs))

        if method in self.failures:
            raise self.failures.pop(method)

        keyed = self.by_args.get(method, {})
        if args in keyed:
            return keyed[args]

        queued = self.responses.get(method)
        if queued:
            return queued.pop(0)

        raise UnconfiguredCall(
            f"{type(self).__name__}.{method}{args} was called with no response, "
            f"by_args entry, or failure configured for it."
        )

    def call_count(self, method: str) -> int:
        return sum(1 for m, *_ in self.calls if m == method)


class FakeSpotifyClient(_AsyncFake):
    """Implements SpotifyClientProtocol."""

    async def artist(self, artist_id: str) -> dict:
        return self._invoke("artist", (artist_id,), {})

    async def artist_top_tracks(self, artist_id: str) -> dict:
        return self._invoke("artist_top_tracks", (artist_id,), {})

    async def track(self, track_id: str) -> dict:
        return self._invoke("track", (track_id,), {})

    async def search(self, query: str, type: str) -> dict:
        return self._invoke("search", (query, type), {})

    async def current_user(self) -> dict:
        return self._invoke("current_user", (), {})

    async def current_user_saved_tracks(self, limit: int) -> dict:
        return self._invoke("current_user_saved_tracks", (limit,), {})

    async def current_user_playlists(self, limit: int) -> dict:
        return self._invoke("current_user_playlists", (limit,), {})

    async def playlist(self, playlist_id: str) -> dict:
        return self._invoke("playlist", (playlist_id,), {})

    async def album_tracks(self, album_id: str, limit: int) -> dict:
        return self._invoke("album_tracks", (album_id, limit), {})

    async def artist_albums(self, artist_id: str, limit: int) -> dict:
        return self._invoke("artist_albums", (artist_id, limit), {})

    async def current_playback(self) -> dict | None:
        return self._invoke("current_playback", (), {})

    async def queue(self) -> dict:
        return self._invoke("queue", (), {})


class FakeMusicBrainzClient(_AsyncFake):
    """Implements MusicBrainzClientProtocol."""

    async def search_recordings(self, query: str, limit: int | None = None) -> dict:
        return self._invoke("search_recordings", (query,), {"limit": limit})

    async def get_artist_by_id(self, artist_id: str, includes: list[str] | None = None) -> dict:
        return self._invoke("get_artist_by_id", (artist_id,), {"includes": includes})

    async def get_recording_by_id(
        self, recording_id: str, includes: list[str] | None = None
    ) -> dict:
        return self._invoke("get_recording_by_id", (recording_id,), {"includes": includes})


class FakeLastFMClient(_AsyncFake):
    """Implements LastFMClientProtocol."""

    async def get_track(self, artist: str, title: str):
        return self._invoke("get_track", (artist, title), {})


class FakeDownloaderClient:
    """Implements DownloaderClientProtocol. Keyed by spotify_id rather than
    method name (unlike the three fakes above) - download.py's
    _download_one always calls search() then download() for the same id in
    sequence, so tests naturally think in terms of "what should happen for
    this id", not "what should this method return on its Nth call".

    - self.candidates[spotify_id]: DownloadCandidate search() should return
      for that id. Absent -> search() returns None (_download_one's handled
      "no song found" path).
    - self.paths[spotify_id]: Path download() should return for that id's
      candidate. Needed for any id that's in .candidates and NOT in
      .failures.
    - self.failures[spotify_id]: exception download() should raise for that
      id's candidate. Use a DOWNLOAD_ERRORS member (LookupError,
      DownloaderError, AudioProviderError) to exercise _download_one's
      handled-failure path (dequeues from every target queue, returns), or
      any other exception to exercise its let-it-propagate path.
    - self.calls: list[("search" | "download", spotify_id)] - what
      download.py's fan-out tests assert on (one search+download per
      distinct spotify_id even when wanted by multiple queues).
    """

    def __init__(self):
        self.candidates: dict[str, DownloadCandidate] = {}
        self.paths: dict[str, Path] = {}
        self.failures: dict[str, Exception] = {}
        self.calls: list[tuple[str, str]] = []

    async def search(self, spotify_id: str) -> DownloadCandidate | None:
        self.calls.append(("search", spotify_id))
        return self.candidates.get(spotify_id)

    async def download(self, candidate: DownloadCandidate) -> Path:
        self.calls.append(("download", candidate.spotify_id))

        if candidate.spotify_id in self.failures:
            raise self.failures[candidate.spotify_id]

        try:
            return self.paths[candidate.spotify_id]
        except KeyError:
            raise UnconfiguredCall(
                f"FakeDownloaderClient.download() called for {candidate.spotify_id!r} "
                f"with no .paths entry or .failures entry configured for it."
            ) from None

    def call_count(self, action: str) -> int:
        return sum(1 for a, _ in self.calls if a == action)


# --- Protocol-conformance check -------------------------------------------
#
# Cheap static check (not a full test) that each Fake* actually implements
# every method its Protocol declares, so a Protocol change doesn't silently
# leave a fake out of sync. Deliberately only checks method *names* are a
# superset, not exact signatures - Protocol's own structural typing can't
# verify argument types at runtime either, and re-implementing that here
# would be its own source of false confidence. Called once from a test in
# unit/test_clients.py (or conftest, at collection time), not on every test.

_PROTOCOL_PAIRS = (
    (FakeSpotifyClient, SpotifyClientProtocol),
    (FakeMusicBrainzClient, MusicBrainzClientProtocol),
    (FakeLastFMClient, LastFMClientProtocol),
    (FakeDownloaderClient, DownloaderClientProtocol),
)


def assert_protocol_conformance() -> None:
    for fake_cls, protocol_cls in _PROTOCOL_PAIRS:
        protocol_methods = {name for name in dir(protocol_cls) if not name.startswith("_")}
        fake_methods = {name for name in dir(fake_cls) if not name.startswith("_")}
        missing = protocol_methods - fake_methods
        assert not missing, (
            f"{fake_cls.__name__} is missing {protocol_cls.__name__} methods: {missing}"
        )
