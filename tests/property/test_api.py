# We never exceeds the spotify rate limit.

# API Tokens are always refreshed before expiration.

# API calls that failes due to networking (not 404 or smth) are handled with exponential backoff.

# API calls that return paginated data are always collected in their entirety.

# ListenChunks of a listen add up to ROUGHLY (networking and floats) ms_played.

# ListenChunks always uphold DB constraints.

# An extra wait from backoff in recently played loop doesn't fuck up the listen item in any way.

# All listen start/end reasons for valid state transitions.

# All artist matching via different APIs result in correct matches or fail gracefully (not found).

# LastFM->Spotify->MusicBrainz conversion maintain transitivity
# (if A->B and B->C, then A should relate to C somehow)
# (if A->B, then B->A should also work)
    # Essentially, all transitions from one API to another should result in equivalent items.
