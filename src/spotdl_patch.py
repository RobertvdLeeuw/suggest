"""
Downloader module, this is where all the downloading pre/post processing happens etc.
"""

import asyncio
import datetime
import json
import logging
import re
import shutil
import sys
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

from yt_dlp.postprocessor.modify_chapters import ModifyChaptersPP
from yt_dlp.postprocessor.sponsorblock import SponsorBlockPP

from spotdl.download.progress_handler import ProgressHandler
from spotdl.providers.audio import (
    AudioProvider,
    BandCamp,
    Piped,
    SoundCloud,
    YouTube,
    YouTubeMusic,
)
from spotdl.providers.lyrics import AzLyrics, Genius, LyricsProvider, MusixMatch, Synced
from spotdl.types.options import DownloaderOptionalOptions, DownloaderOptions
from spotdl.types.song import Song
from spotdl.utils.archive import Archive
from spotdl.utils.config import (
    DOWNLOADER_OPTIONS,
    GlobalConfig,
    create_settings_type,
    get_errors_path,
    get_temp_path,
    modernize_settings,
)
from spotdl.utils.ffmpeg import FFmpegError, convert, get_ffmpeg_path
from spotdl.utils.formatter import create_file_name
from spotdl.utils.lrc import generate_lrc
from spotdl.utils.m3u import gen_m3u_files
from spotdl.utils.metadata import MetadataError, embed_metadata
from spotdl.utils.search import gather_known_songs, reinit_song, songs_from_albums

__all__ = [
    "AUDIO_PROVIDERS",
    "LYRICS_PROVIDERS",
    "Downloader",
    "DownloaderError",
    "SPONSOR_BLOCK_CATEGORIES",
]

AUDIO_PROVIDERS: Dict[str, Type[AudioProvider]] = {
    "youtube": YouTube,
    "youtube-music": YouTubeMusic,
    "soundcloud": SoundCloud,
    "bandcamp": BandCamp,
    "piped": Piped,
}

LYRICS_PROVIDERS: Dict[str, Type[LyricsProvider]] = {
    "genius": Genius,
    "musixmatch": MusixMatch,
    "azlyrics": AzLyrics,
    "synced": Synced,
}

SPONSOR_BLOCK_CATEGORIES = {
    "sponsor": "Sponsor",
    "intro": "Intermission/Intro Animation",
    "outro": "Endcards/Credits",
    "selfpromo": "Unpaid/Self Promotion",
    "preview": "Preview/Recap",
    "filler": "Filler Tangent",
    "interaction": "Interaction Reminder",
    "music_offtopic": "Non-Music Section",
}


logger = logging.getLogger("SpotDL")


class DownloaderError(Exception):
    """
    Base class for all exceptions related to downloaders.
    """

class ProgressTrackerDummy:
    yt_dlp_progress_hook = None
    ffmpeg_progress_hook = None

    def notify_complete(*args, **kwargs): pass
    def notify_download_skip(*args, **kwargs): pass
    def notify_download_complete(*args, **kwargs): pass
    def notify_conversion_complete(*args, **kwargs): pass
    def notify_error(*args, **kwargs): pass

class ProgressHandlerDummy:
    def set_song_count(self, *args, **kwargs): pass
    def get_new_tracker(self, *args, **kwargs): return ProgressTrackerDummy()

def _patched_init(
    self,
    settings: Optional[Union[DownloaderOptionalOptions, DownloaderOptions]] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
):
    """
    Initialize the Downloader class.

    ### Arguments
    - settings: The settings to use.
    - loop: The event loop to use.

    ### Notes
    - `search-query` uses the same format as `output`.
    - if `audio_provider` or `lyrics_provider` is a list, then if no match is found,
        the next provider in the list will be used.
    """

    if settings is None:
        settings = {}

    # Create settings dictionary, fill in missing values with defaults
    # from spotdl.types.options.DOWNLOADER_OPTIONS
    self.settings: DownloaderOptions = DownloaderOptions(
        **create_settings_type(
            Namespace(config=False), dict(settings), DOWNLOADER_OPTIONS
        )  # type: ignore
    )

    # Handle deprecated values in config file
    modernize_settings(self.settings)
    logger.debug("Downloader settings: %s", self.settings)

    # If no audio providers specified, raise an error
    if len(self.settings["audio_providers"]) == 0:
        raise DownloaderError(
            "No audio providers specified. Please specify at least one."
        )

    # If ffmpeg is the default value and it's not installed
    # try to use the spotdl's ffmpeg
    self.ffmpeg = self.settings["ffmpeg"]
    if self.ffmpeg == "ffmpeg" and shutil.which("ffmpeg") is None:
        ffmpeg_exec = get_ffmpeg_path()
        if ffmpeg_exec is None:
            raise DownloaderError("ffmpeg is not installed")

        self.ffmpeg = str(ffmpeg_exec.absolute())

    logger.debug("FFmpeg path: %s", self.ffmpeg)

    self.loop = loop or (
        asyncio.new_event_loop()
        if sys.platform != "win32"
        else asyncio.ProactorEventLoop()  # type: ignore
    )

    if loop is None:
        asyncio.set_event_loop(self.loop)

    # semaphore is required to limit concurrent asyncio executions
    self.semaphore = asyncio.Semaphore(self.settings["threads"])

    self.progress_handler = ProgressHandlerDummy()

    # Gather already present songs
    self.scan_formats = self.settings["detect_formats"] or [self.settings["format"]]
    self.known_songs: Dict[str, List[Path]] = {}
    if self.settings["scan_for_songs"]:
        logger.info("Scanning for known songs, this might take a while...")
        for scan_format in self.scan_formats:
            logger.debug("Scanning for %s files", scan_format)

            found_files = gather_known_songs(self.settings["output"], scan_format)

            logger.debug("Found %s %s files", len(found_files), scan_format)

            for song_url, song_paths in found_files.items():
                known_paths = self.known_songs.get(song_url)
                if known_paths is None:
                    self.known_songs[song_url] = song_paths
                else:
                    self.known_songs[song_url].extend(song_paths)

    logger.debug("Found %s known songs", len(self.known_songs))

    # Initialize lyrics providers
    self.lyrics_providers: List[LyricsProvider] = []
    for lyrics_provider in self.settings["lyrics_providers"]:
        lyrics_class = LYRICS_PROVIDERS.get(lyrics_provider)
        if lyrics_class is None:
            raise DownloaderError(f"Invalid lyrics provider: {lyrics_provider}")
        if lyrics_provider == "genius":
            access_token = self.settings.get("genius_token")
            if not access_token:
                raise DownloaderError("Genius token not found in settings")
            self.lyrics_providers.append(Genius(access_token))
        else:
            self.lyrics_providers.append(lyrics_class())

    # Initialize audio providers
    self.audio_providers: List[AudioProvider] = []
    for audio_provider in self.settings["audio_providers"]:
        audio_class = AUDIO_PROVIDERS.get(audio_provider)
        if audio_class is None:
            raise DownloaderError(f"Invalid audio provider: {audio_provider}")

        self.audio_providers.append(
            audio_class(
                output_format=self.settings["format"],
                cookie_file=self.settings["cookie_file"],
                search_query=self.settings["search_query"],
                filter_results=self.settings["filter_results"],
                yt_dlp_args=self.settings["yt_dlp_args"],
            )
        )

    # Initialize list of errors
    self.errors: List[str] = []

    # Initialize proxy server
    proxy = self.settings["proxy"]
    proxies = None
    if proxy:
        if not re.match(
            pattern=r"^(http|https):\/\/(?:(\w+)(?::(\w+))?@)?((?:\d{1,3})(?:\.\d{1,3}){3})(?::(\d{1,5}))?$",  # pylint: disable=C0301
            string=proxy,
        ):
            raise DownloaderError(f"Invalid proxy server: {proxy}")
        proxies = {"http": proxy, "https": proxy}
        logger.info("Setting proxy server: %s", proxy)

    GlobalConfig.set_parameter("proxies", proxies)

    # Initialize archive
    self.url_archive = Archive()
    if self.settings["archive"]:
        self.url_archive.load(self.settings["archive"])

    logger.debug("Archive: %d urls", len(self.url_archive))

    logger.debug("Downloader initialized")


from spotdl.download.downloader import Downloader
Downloader.__init__ = _patched_init
