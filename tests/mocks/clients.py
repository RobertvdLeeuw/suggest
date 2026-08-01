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

# FakeSpotifyClient: implements SpotifyClientProtocol.
# needs:
#   - per-test configurable returns/exceptions keyed by (method_name, args) or
#     just method_name if args don't matter for a given test.
#   - a .calls: list[tuple[str, tuple, dict]] call log - services.py's
#     "doesn't call resolve when repo already has it" ordering assertions
#     need call-count/call-presence checks on these fakes, not just return values.
# touches: collecter.clients.SpotifyClientProtocol (structural conformance)

# FakeMusicBrainzClient: implements MusicBrainzClientProtocol. Same shape as above.
# touches: collecter.clients.MusicBrainzClientProtocol

# FakeLastFMClient: implements LastFMClientProtocol. Same shape as above.
# touches: collecter.clients.LastFMClientProtocol

# FakeDownloaderClient: implements DownloaderClientProtocol - .search()/.download()
# configurable per spotify_id via .candidates/.failures/.paths dicts (this part of
# the pre-refactor shape was already reasonable, worth keeping the interface).
# needs a .calls log too, for download.py's fan-out assertions (one search+download
# per distinct spotify_id, even when multiple queues want it).
# touches: collecter.clients.downloader.DownloaderClientProtocol,
#          collecter.clients.downloader.DownloadCandidate,
#          collecter.clients.downloader.DOWNLOAD_ERRORS

# Protocol-conformance check: worth a cheap static assertion (not a full test)
# that each Fake* actually satisfies its Protocol's method signatures, so a
# Protocol change doesn't silently leave a fake out of sync.
# touches: typing.get_type_hints or similar, one assert per Fake*/Protocol pair
