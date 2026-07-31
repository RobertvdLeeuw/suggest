"""
Purpose & scope
---------------
Pure translation between resolution.py's dataclasses (ResolvedArtist,
ResolvedTrack) and the ORM rows in models.py (Artist, Song, ArtistMetadata,
SongMetadata). No I/O, no session, no network calls - if it needs a session
it belongs in repository.py instead.

This is a one-way street in normal operation: ResolvedArtist/Track goes in,
an (unpersisted) ORM object comes out. repository.py is what actually saves
it and hands back the canonical, DB-backed row.
"""

from models import Artist, ArtistMetadata, MetadataType, Song, SongMetadata

from .resolution import ResolvedArtist, ResolvedTrack

# resolution.py's tags dict keys ("lastfm", "musicbrainz") -> the `source`
# string stored on ArtistMetadata/SongMetadata rows. Kept here rather than in
# resolution.py since it's purely a mapping/display concern - resolution.py
# shouldn't need to know how its output gets labeled in the DB.
_SOURCE_LABELS = {
    "lastfm": "LastFM",
    "musicbrainz": "MusicBrainz",
}


def _tags_to_metadata(tags: dict[str, list[str]], metadata_cls):
    """Flattens a resolution.py tags dict (source -> tag list) into a flat list of
    ArtistMetadata/SongMetadata rows. Every tag currently maps to
    MetadataType.genre - resolution.py doesn't distinguish tags from genres, and
    neither did the code this replaces. Rows are unpersisted and carry no
    artist_id/song_id yet - repository.py fills that in once the parent row
    exists."""
    return [
        metadata_cls(
            type=MetadataType.genre,
            value=value,
            source=_SOURCE_LABELS.get(source, source),
        )
        for source, values in tags.items()
        for value in values
    ]


def artist_to_orm(resolved: ResolvedArtist) -> Artist:
    """Builds an (unpersisted) Artist + its ArtistMetadata rows from a ResolvedArtist.
    The metadata rows are attached via Artist.extra_data - repository.py reads them
    off there (as the `metadata` argument to create_artist), it never builds them
    itself."""
    artist = Artist(spotify_id=resolved.spotify_id, artist_name=resolved.name)
    artist.extra_data = _tags_to_metadata(resolved.tags, ArtistMetadata)
    return artist


def track_to_orm(resolved: ResolvedTrack, artists: list[Artist]) -> Song:
    """Builds an (unpersisted) Song + its SongMetadata rows from a ResolvedTrack.
    `artists` are the already-persisted Artist rows for resolved.artists, in order -
    resolving/persisting artists is repository.py's job, not this function's. The
    metadata rows are attached via Song.extra_data, same convention as
    artist_to_orm."""
    song = Song(spotify_id=resolved.spotify_id, song_name=resolved.name, artists=artists)
    song.extra_data = _tags_to_metadata(resolved.tags, SongMetadata)
    return song
