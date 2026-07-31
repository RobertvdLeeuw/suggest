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

from models import Artist, ArtistMetadata, Song, SongMetadata

from .resolution import ResolvedArtist, ResolvedTrack


def artist_to_orm(resolved: ResolvedArtist) -> Artist:
    """Builds an (unpersisted) Artist + its ArtistMetadata rows from a ResolvedArtist."""
    ...


def track_to_orm(resolved: ResolvedTrack, artists: list[Artist]) -> Song:
    """Builds an (unpersisted) Song + its SongMetadata rows from a ResolvedTrack.
    `artists` are the already-persisted Artist rows for resolved.artists, in order -
    resolving/persisting artists is repository.py's job, not this function's."""
    ...
