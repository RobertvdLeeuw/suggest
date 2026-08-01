"""
Raw musicbrainzngs-shaped double - what test_clients.py's musicbrainz module
tests call into (search_recordings/get_artist_by_id/get_recording_by_id and
the rate-limit/backoff wrapping around them). Same layer distinction as
raw_spotify.py: this fakes the third-party library, not our Protocol.

Old apis.py's musicbrainz_fake injected random failures inline
(random.random() < 0.01: raise ResponseError). Don't carry that over -
failure injection should be explicit per-test (via .failures below) so a
failing test is reproducible, not something that occasionally passes by luck.
"""

# RawMusicBrainz: configurable double for the musicbrainzngs module-level API
# (mb.search_recordings, mb.get_artist_by_id, mb.get_recording_by_id,
# mb.set_useragent, mb.auth).
#
# needs:
#   - default returns from fixtures.api_responses.MB_SEARCH_RECORDINGS,
#     MB_ARTIST_BY_ID, MB_RECORDING_BY_ID.
#   - .failures[method_name] = exception to raise on next call - needs both
#     shapes _classify cares about: mb.ResponseError with .cause.code == 503
#     (rate limit, retryable) and a plain 4xx-shaped ResponseError (fatal).
#   - a call log, same purpose as raw_spotify.py's - clients/musicbrainz.py's
#     _throttled_to_thread rate-limiting (max 1 req/s via the module Semaphore)
#     needs call *timing* observable too, not just call count, for the "never
#     exceeds rate limit" assertion.
#   - configure()/auth() should be no-ops that just record they were called,
#     for the "must call configure() before use" assert check.
# touches: fixtures.api_responses (MB_SEARCH_RECORDINGS, MB_ARTIST_BY_ID,
#          MB_RECORDING_BY_ID), musicbrainzngs.ResponseError, musicbrainzngs.NetworkError
