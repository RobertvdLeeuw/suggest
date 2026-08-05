# All artist matching via different APIs result in correct matches or fail gracefully (not found).
# touches: collecter.resolution.resolve_artist, collecter.resolution.ResolvedArtist,
#          mocks.clients.FakeSpotifyClient/FakeMusicBrainzClient/FakeLastFMClient,
#          strategies.resolution.resolved_artist_strat, strategies.resolution.client_failure_sequence_strat

# LastFM->Spotify->MusicBrainz conversion maintain transitivity
# (if A->B and B->C, then A should relate to C somehow)
# (if A->B, then B->A should also work)
# Essentially, all transitions from one API to another should result in equivalent items.
# touches: collecter.resolution.resolve_artist, collecter.resolution._sp_artist_to_lastfm,
#          collecter.resolution._lastfm_to_sp, collecter.resolution._sp_artist_to_mb,
#          mocks.clients.FakeSpotifyClient/FakeMusicBrainzClient/FakeLastFMClient

# An unavailable source never leaks into tags - a transient failure is never mistaken
# for a confident "no match" or persisted as a real tag.
# touches: collecter.resolution.resolve_artist, collecter.resolution.resolve_track,
#          collecter.resolution.UNAVAILABLE, collecter.resolution.ResolvedArtist.unavailable,
#          strategies.resolution.client_failure_sequence_strat

# resolve_artist/resolve_track never raise on a client failure - always return a
# result object with the failing source(s) recorded in `unavailable` instead.
# touches: collecter.resolution.resolve_artist, collecter.resolution.resolve_track,
#          mocks.clients (fakes configured to raise per-source)

# If every known source is unavailable, tags is empty and unavailable == KNOWN_SOURCES
# (nothing fabricated when nothing could be checked).
# touches: collecter.resolution.resolve_artist, collecter.resolution.KNOWN_SOURCES

# get_similar_artists never returns an artist unavailable/errored source silently as "similar".
# touches: collecter.resolution.get_similar_artists
