"""
Purpose & scope
---------------
Pure cross-referencing logic between Spotify, MusicBrainz, and LastFM. Takes
a seed ID (usually Spotify's) and the three client Protocols, and returns a
ResolvedArtist/ResolvedTrack - plain data, no DB, no ORM.

Rules for this file:
  - Never import sqlalchemy, models, or repository.py.
  - Never hold onto or return a DB session.
  - Clients are always passed in as arguments (typed against the Protocols in
    clients/__init__.py), never imported as module-level singletons - that's
    what makes these functions testable with fakes and no network.
  - ResolvedArtist/ResolvedTrack stay minimal: only what resolution actually
    produces (name, cross-service ids, tags). No DB-state fields
    (artist_id, entirely_queued, timestamps, etc.) ever get added here -
    those belong to the ORM row that repository.py/mapping.py produce, and
    that row - not this dataclass - is the canonical reference downstream
    code should hold onto.
"""

from dataclasses import dataclass, field

from .clients import SpotifyClientProtocol, MusicBrainzClientProtocol, LastFMClientProtocol


@dataclass
class ResolvedArtist:
    name: str
    spotify_id: str | None = None
    musicbrainz_id: str | None = None
    lastfm_name: str | None = None
    tags: dict[str, list[str]] = field(default_factory=dict)  # source -> tags


@dataclass
class ResolvedTrack:
    name: str
    spotify_id: str
    artists: list[ResolvedArtist]
    tags: dict[str, list[str]] = field(default_factory=dict)  # source -> tags


def resolve_artist(
    spotify_id: str,
    spotify: SpotifyClientProtocol,
    musicbrainz: MusicBrainzClientProtocol,
    lastfm: LastFMClientProtocol,
) -> ResolvedArtist:
    """Cross-reference a Spotify artist against MusicBrainz and LastFM, collecting tags
    from whichever sources produce a confident match. Never raises on a source having
    no match - that source's fields are simply left None/empty."""
    ...


def resolve_track(
    spotify_id: str,
    spotify: SpotifyClientProtocol,
    musicbrainz: MusicBrainzClientProtocol,
    lastfm: LastFMClientProtocol,
) -> ResolvedTrack:
    """Cross-reference a Spotify track (and its artists) against MusicBrainz and LastFM."""
    ...


def get_similar_artists(
    spotify_artist_id: str,
    spotify: SpotifyClientProtocol,
    lastfm: LastFMClientProtocol,
    degrees: int = 1,
) -> list[str]:
    """Returns Spotify artist ids of artists similar to the given one, via LastFM's
    similarity graph, translated back to Spotify ids."""
    ...
