# We never exceed the Spotify rate limit.
# touches: collecter.clients.spotify.SpotifyClient, collecter.clients.spotify._classify,
#          mocks.raw_spotify.RawSpotify (injecting 429 + Retry-After)

# API Tokens are always refreshed before expiration.
# touches: collecter.clients.spotify.SpotifyClient (via spotipy.oauth2.SpotifyOAuth)
# QUESTIONABLE: spotipy's SpotifyOAuth handles refresh internally - unclear whether
# there's any of our own code left to actually test here, or whether this is really
# spotipy's contract to uphold, not ours. Confirm before writing real assertions;
# may end up deleted rather than implemented.

# API calls that return paginated data are always collected in their entirety.
# touches: collecter.clients.spotify.SpotifyClient (methods that loop page["next"]),
#          mocks.raw_spotify.RawSpotify, strategies.apis.spotify_paginated_response_strat

# MusicBrainz calls never exceed its documented 1 req/s rate limit, regardless of how
# many coroutines call in concurrently.
# touches: collecter.clients.musicbrainz._throttled_to_thread, collecter.clients.musicbrainz._rate_limit,
#          mocks.raw_musicbrainz.RawMusicBrainz (call-timing log)

# MusicBrainz's 503-wrapped-in-ResponseError is classified retryable; a well-formed
# 4xx-shaped ResponseError is classified fatal.
# touches: collecter.clients.musicbrainz._classify, mocks.raw_musicbrainz.RawMusicBrainz

# LastFM's WSError id 29 (rate limit) is classified retryable; any other WSError is fatal.
# touches: collecter.clients.lastfm.classify_lastfm, mocks.raw_lastfm

# musicbrainz.search_recordings/get_artist_by_id/get_recording_by_id all assert
# _configured before use - calling before configure() fails loudly, not silently.
# touches: collecter.clients.musicbrainz.configure, collecter.clients.musicbrainz._configured
