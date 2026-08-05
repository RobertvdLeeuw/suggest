"""
Raw pylast-shaped double - what test_clients.py's LastFMClient tests call
into, AND what resolution.py's LastFM-chain tests (_lastfm_to_sp,
_lastfm_top_tags etc.) need for the pylast.Track/pylast.Artist-like objects
those helpers call .get_artist()/.get_top_tags()/.get_similar()/
.get_top_tracks() on directly (not through LastFMClientProtocol - see
resolution.py's module comment on this).
"""

# RawLastFMNetwork: double for pylast.LastFMNetwork - only needs get_track()
# and enable_rate_limit() since that's all clients/lastfm.py's LastFMClient
# touches on it.
# touches: pylast.LastFMNetwork's public surface used by clients/lastfm.py

# RawTrack / RawArtist: doubles for the pylast.Track/pylast.Artist objects
# resolution.py's LastFM chain calls directly (get_artist, get_top_tags,
# get_similar, get_top_tracks) - these are NOT LastFMClientProtocol methods,
# they're plain pylast object methods chained off LastFMClient.get_track()'s
# result, so they need their own thin doubles rather than reusing
# clients.FakeLastFMClient.
#
# needs:
#   - .get_top_tags()/.get_similar()/.get_top_tracks() returning
#     pylast.TopItem-like objects wrapping name-bearing items (a NamedItem
#     shape, same idea as the old apis.py's NamedItem/TopItem usage).
#   - .failures configuration raising pylast.WSError with a settable
#     .get_id() - id 29 (STATUS_RATE_LIMIT_EXCEEDED) is the retryable case
#     _classify_lastfm_failure/classify_lastfm both branch on, anything else
#     is fatal/confident-miss.
# touches: pylast.WSError, pylast.TopItem, fixtures.api_responses (no LastFM
#          fixtures exist yet - old apis.py's Track_fake/Artist_fake tags
#          were synthetic (Funky/Groovy/Large/Non-fiction, not real API
#          captures) - worth deciding whether these need real captured
#          LastFM shapes added to fixtures.api_responses or synthetic is fine
#          here specifically since pylast wraps XML into these objects for us
#          already, there's no raw JSON shape to be faithful to.
