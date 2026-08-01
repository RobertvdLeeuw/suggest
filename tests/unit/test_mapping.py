# artist_to_orm/track_to_orm never drop or mutate fields present on the ResolvedArtist/Track.
# touches: collecter.mapping.artist_to_orm, collecter.mapping.track_to_orm,
#          strategies.resolution.resolved_artist_strat, strategies.resolution.resolved_track_strat

# artist_to_orm/track_to_orm are total: every valid ResolvedArtist/Track produces a
# valid ORM row (no exceptions on well-formed input).
# touches: collecter.mapping.artist_to_orm, collecter.mapping.track_to_orm,
#          models.Artist, models.Song

# Every tag in a ResolvedArtist/Track's `tags` dict produces exactly one metadata row
# (no drops, no duplicates) - source->tag-list flattening in _tags_to_metadata is lossless.
# touches: collecter.mapping._tags_to_metadata, models.ArtistMetadata, models.SongMetadata

# Every metadata row's `source` label is a recognized display label (_SOURCE_LABELS
# mapping, or the raw source name as fallback) - never empty, never leaks the raw
# resolution.py source key when a display label exists for it.
# touches: collecter.mapping._SOURCE_LABELS

# track_to_orm's `artists` list on the resulting Song matches the `artists` argument
# passed in, in order - mapping.py never re-derives artist order from resolved.artists
# independently of what repository.py/services.py already resolved.
# touches: collecter.mapping.track_to_orm

# Every produced ArtistMetadata/SongMetadata row's type is MetadataType.genre (current
# mapping.py behavior - resolution.py doesn't yet distinguish tags from genres). Flagged
# as worth revisiting if that distinction ever gets added upstream, but the current
# contract is a fixed type, so it should be asserted, not assumed.
# touches: collecter.mapping._tags_to_metadata, models.MetadataType
