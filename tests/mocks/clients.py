"""Fake implementations of the SpotifyClientProtocol/MusicBrainzClientProtocol/
LastFMClientProtocol trio, for tests that need resolution.py or services.py
behavior without any real network call. Configurable per-test rather than
random, unlike the old tests/mocks/apis.py fakes - hypothesis strategies drive
the randomness instead, these just need to return what they're told to."""

from collecter.clients import SpotifyClientProtocol, MusicBrainzClientProtocol, LastFMClientProtocol
from collecter.clients.downloader import DownloadCandidate, DownloaderClientProtocol


class FakeSpotifyClient:
    def artist(self, artist_id: str) -> dict: ...
    def artist_top_tracks(self, artist_id: str) -> dict: ...
    def track(self, track_id: str) -> dict: ...
    def search(self, query: str, type: str) -> dict: ...
    def current_user(self) -> dict: ...
    def current_user_saved_tracks(self, limit: int) -> dict: ...
    def current_user_playlists(self, limit: int) -> dict: ...
    def playlist(self, playlist_id: str) -> dict: ...
    def album_tracks(self, album_id: str, limit: int) -> dict: ...
    def artist_albums(self, artist_id: str, limit: int) -> dict: ...
    def current_playback(self) -> dict | None: ...
    def queue(self) -> dict: ...


class FakeMusicBrainzClient:
    def search_recordings(self, query: str) -> dict: ...
    def get_artist_by_id(self, artist_id: str, includes: list[str]) -> dict: ...
    def get_recording_by_id(self, recording_id: str, includes: list[str]) -> dict: ...


class FakeLastFMClient:
    def get_track(self, artist: str, title: str): ...


class FakeDownloaderClient:
    """Configurable per-test fake for DownloaderClientProtocol - set
    .candidates[spotify_id] / .failures[spotify_id] / .paths[spotify_id]
    before use; download.py's orchestration tests need this, not a real
    spotdl call."""

    def __init__(self):
        self.candidates: dict[str, DownloadCandidate | None] = {}
        self.failures: dict[str, Exception] = {}
        self.paths: dict[str, str] = {}

    async def search(self, spotify_id: str) -> DownloadCandidate | None:
        return self.candidates.get(
            spotify_id, DownloadCandidate(spotify_id, "Test Song", "Test Artist", _song=None)
        )

    async def download(self, candidate: DownloadCandidate):
        from pathlib import Path

        if candidate.spotify_id in self.failures:
            raise self.failures[candidate.spotify_id]
        return Path(self.paths.get(candidate.spotify_id, f"/downloads/{candidate.spotify_id}.wav"))
