"""
Strategies at the resolution.py domain level - ResolvedArtist/ResolvedTrack
and the tags/unavailable shapes they carry. Operate on the already-parsed
domain objects, not raw API JSON (that's strategies/apis.py's job) - these
feed resolution.py-level tests and mapping.py tests directly via
mocks/clients.py's fakes, not via mocks/raw_*.py.
"""

# resolved_artist_strat: ResolvedArtist with random name (via
# strategies.apis.unicode_name_strat), optional spotify_id/musicbrainz_id/
# lastfm_name, tags dict keyed from a subset of resolution.KNOWN_SOURCES,
# and unavailable as a random subset of KNOWN_SOURCES disjoint from tags'
# keys (a source can't be both resolved and unavailable at once - that's the
# invariant resolution.py's docstring describes, the strategy should never
# generate a state resolution.py itself would consider invalid).
# touches: collecter.resolution.ResolvedArtist, collecter.resolution.KNOWN_SOURCES

# resolved_track_strat: ResolvedTrack, artists=list of resolved_artist_strat()
# (0 to a few), same tags/unavailable shape as above.
# touches: collecter.resolution.ResolvedTrack

# tags_dict_strat: source -> list[str] dict, keys from KNOWN_SOURCES, values
# from strategies.apis.unicode_name_strat (tag names hit the same
# unicode/empty-string edge cases artist names do).
# touches: collecter.resolution.KNOWN_SOURCES

# client_failure_sequence_strat: for resolution.py's resolve_artist/
# resolve_track property tests - draws which of the 3 clients
# (spotify/musicbrainz/lastfm) fail vs succeed on a given resolution attempt,
# to drive mocks.clients fakes into every combination of confident-miss vs
# UNAVAILABLE vs success. This is what actually exercises the "unavailable
# vs confidently empty" distinction resolution.py's module docstring is
# built around.
# touches: collecter.resolution.UNAVAILABLE (the sentinel, for confirming a
#          strategy-driven failure surfaces as this specific value downstream)
