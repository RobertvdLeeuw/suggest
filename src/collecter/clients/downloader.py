"""
Purpose & scope
---------------
Real implementation of DownloaderClientProtocol - wraps spotdl to search a
Spotify track id and download the matched audio to disk. Same convention as
clients/spotify.py: spotdl (and the spotipy client it needs internally) is
synchronous, so every call here runs via asyncio.to_thread - callers always
await it, and a slow/blocking download no longer stalls the whole event loop
the way old/downloader.py's direct spotdl.search()/spotdl.download() calls
did (that call sat inline in an async function with nothing between it and
the loop).

Owns its own spotipy.Spotify instance (token-refreshed via SpotifyOAuth) -
spotdl's constructor wants a raw spotipy.Spotify, not the
SpotifyClientProtocol the rest of collecter/ codes against, so this file
builds its own rather than borrowing clients/spotify.py's internal one.
Same env var names as old/metadata.py's module-level `sp`/`sp_oauth`
(SPOTIFY_CLIENT_ID/SECRET, SPOTIFY_REFRESH_TOKEN) - existing .env keeps
working.

Does NOT touch the DB, the embedding queues, or the filesystem beyond
handing back the path spotdl wrote to - renaming/moving the file, deciding
which queues get it, and cleaning up the downloads folder is download.py's
job, not this file's.

NOTE: `loop=asyncio.get_event_loop()` was passed to Spotdl() in the old
implementation. Whether spotdl needs a loop reference at all, and whether
one grabbed off the *calling* thread (rather than the to_thread worker) is
still correct, depends on spotdl's internals - worth confirming against the
installed spotdl version rather than assuming this port is a like-for-like
behavioral match; flagging rather than guessing.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import spotipy
from spotdl import Spotdl
from spotdl.download.downloader import DownloaderError
from spotdl.providers.audio.base import AudioProviderError
from spotdl.types.options import DownloaderOptions
from spotipy.oauth2 import SpotifyOAuth

LOGGER = logging.getLogger(__name__)

# Failures spotdl itself raises for "couldn't get this track" - expected,
# handled by download.py (dequeue + move on). Anything else propagates as a
# genuine bug rather than being swallowed, unlike old/downloader.py's bare
# `except Exception`.
DOWNLOAD_ERRORS = (LookupError, DownloaderError, AudioProviderError)


@dataclass(frozen=True)
class DownloadCandidate:
    """Result of a successful search() - name/artist are for logging only.
    _song is an opaque spotdl.types.song.Song handle that only download()
    needs; nothing else in collecter/ should inspect it."""

    spotify_id: str
    name: str
    artist: str
    _song: object


class DownloaderClientProtocol(Protocol):
    async def search(self, spotify_id: str) -> DownloadCandidate | None: ...
    async def download(self, candidate: DownloadCandidate) -> Path: ...


class SpotdlDownloaderClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        download_dir: str,
        redirect_uri: str = "http://127.0.0.1:8888/callback",
    ):
        token_info = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
        ).refresh_access_token(refresh_token)

        self._spotdl = Spotdl(
            no_cache=True,
            spotify_client=spotipy.Spotify(auth=token_info["access_token"]),
            downloader_settings=DownloaderOptions(
                format="wav",
                simple_tui=False,
                print_download_errors=True,
                output=download_dir,
                overwrite="skip",
            ),
        )

    async def search(self, spotify_id: str) -> DownloadCandidate | None:
        songs = await asyncio.to_thread(
            self._spotdl.search, [f"https://open.spotify.com/track/{spotify_id}"]
        )
        if not songs:
            return None

        song = songs[0]
        return DownloadCandidate(
            spotify_id=spotify_id, name=song.name, artist=song.artist, _song=song
        )

    async def download(self, candidate: DownloadCandidate) -> Path:
        """Raises a DOWNLOAD_ERRORS member on a handled failure - download.py
        is what catches those, never this module."""
        _, file_path = await asyncio.to_thread(self._spotdl.download, candidate._song)

        if not file_path or not Path(file_path).exists():
            raise DownloaderError(f"Download failed or file not found: {file_path}")

        return Path(file_path)
