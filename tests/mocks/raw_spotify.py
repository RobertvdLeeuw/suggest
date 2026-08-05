"""
Raw spotipy.Spotify-shaped double - what test_clients.py's SpotifyClient
tests call into. NOT the same layer as clients.FakeSpotifyClient: this one
stands in for the third-party library itself (spotipy), so
clients/spotify.py's real pagination/backoff/_classify code runs against it
unmodified. clients.FakeSpotifyClient (in this same dir) stands in for OUR
SpotifyClientProtocol instead, one layer up - resolution.py/services.py
tests use that one, never this one.

Default responses should come from fixtures/api_responses.py's
SPOTIFY_* constants - real captured shapes, not invented ones. Per-test
configuration (raising, paginating, going stale) should override those
defaults per-method, not by hand-editing the fixtures.
"""

# RawSpotify: configurable double for the raw spotipy.Spotify client
# clients/spotify.py's SpotifyClient wraps.
#
# needs:
#   - one method per SpotifyClient wrapper method (artist, artist_top_tracks,
#     track, search, current_user, current_user_saved_tracks,
#     current_user_playlists, playlist, album_tracks, artist_albums,
#     current_playback, queue) - default return pulled from the matching
#     fixtures.api_responses.SPOTIFY_* constant.
#   - per-method override hooks: set .responses[method_name] = [resp1, resp2, ...]
#     to return a queue of values across calls (pagination), or
#     .failures[method_name] = SpotifyException(...) to raise on next call.
#   - pagination support: needs to honor 'next' pages the way the real
#     paginated SpotifyClient methods walk them - configurable page count/size
#     via strategies.apis rather than hardcoded here.
#   - a call log (list of (method_name, args, kwargs)) for tests asserting on
#     call counts (rate-limit / backoff retry-count assertions).
# touches: fixtures.api_responses (SPOTIFY_ARTIST, SPOTIFY_ARTIST_TOP_TRACKS,
#          SPOTIFY_TRACK, SPOTIFY_SEARCH, SPOTIFY_CURRENT_USER,
#          SPOTIFY_SAVED_TRACKS, SPOTIFY_CURRENT_USER_PLAYLISTS,
#          SPOTIFY_PLAYLIST, SPOTIFY_ALBUM_TRACKS, SPOTIFY_ARTIST_ALBUMS,
#          SPOTIFY_CURRENT_PLAYBACK, SPOTIFY_QUEUE),
#          spotipy.exceptions.SpotifyException (for injected 429/5xx/4xx)

# (Dropped: RawSpotifyOAuth. Confirmed token-refresh is entirely spotipy's own
# contract to uphold, not collecter's - see test_clients.py.)
