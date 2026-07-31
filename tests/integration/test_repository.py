# Database sessions always commit or rollback cleanly (never left in intermediate state)

# All inserts that don't follow contraints are failed gracefully.
    # This one needs to be fleshed way the fuck out.

# Database connection pool never gets exhausted under high load
# (test your pool_size=10, max_overflow=20 settings)

# Constraint violations during bulk operations don't corrupt successfully inserted data
# (test partial failure scenarios in your batch operations)

# get_or_create is race-safe, for every model that uses it (Artist, Song, User,
# ArtistMetadata, SongMetadata via get_or_create_many): every time 2 processes try to
# create the same object concurrently, their contents match, only 1 instance is pushed
# to db, and both return a valid reference to that single instance.
    # Parametrize across models instead of one-off tests per entity.

# System recovers gracefully from DB outage.

# ListenChunks always uphold DB constraints.
