# (No PBT assertions from the old files targeted mapping.py specifically - it's new
#  surface area from the refactor. Worth writing e.g.:)
# artist_to_orm/track_to_orm never drop or mutate fields present on the ResolvedArtist/Track.
# artist_to_orm/track_to_orm are total: every valid ResolvedArtist/Track produces a valid ORM row (no exceptions on well-formed input).
