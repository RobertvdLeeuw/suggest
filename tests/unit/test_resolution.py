# All artist matching via different APIs result in correct matches or fail gracefully (not found).

# LastFM->Spotify->MusicBrainz conversion maintain transitivity
# (if A->B and B->C, then A should relate to C somehow)
# (if A->B, then B->A should also work)
    # Essentially, all transitions from one API to another should result in equivalent items.
