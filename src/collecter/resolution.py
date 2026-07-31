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

unavailable vs "confidently empty"
-----------------------------------
Every cross-reference can fail two different ways: the source was checked and
genuinely has nothing (a real "no match"), or the source couldn't be checked
at all (network error, rate limit exhausted after clients/retry.py's own
retries, service down for maintenance). Those are NOT the same outcome - the
first is permanent, the second is worth trying again later. resolve_artist/
resolve_track report the second case via `unavailable: set[str]` (which
source names, e.g. {"lastfm"}), kept separate from `tags` so a transient
failure can never be mistaken for - or persisted as - a real tag by
mapping.py. services.py is what actually decides what to do about an
unavailable source (see PendingArtistMetadata/PendingSongMetadata in
models.py, and services.retry_pending_metadata).
"""

import asyncio
from dataclasses import dataclass, field

import pylast

from .clients import SpotifyClientProtocol, MusicBrainzClientProtocol, LastFMClientProtocol

KNOWN_SOURCES = frozenset({"lastfm", "musicbrainz"})


class _Unavailable:
    """Sentinel distinguishing 'checked, source has nothing' (None/[]) from
    'could not check the source right now' (this). Not an exception - resolution
    functions never raise on a source failure, they report it in the result."""

    def __repr__(self):
        return "UNAVAILABLE"


UNAVAILABLE = _Unavailable()


@dataclass
class ResolvedArtist:
    name: str
    spotify_id: str | None = None
    musicbrainz_id: str | None = None
    lastfm_name: str | None = None
    tags: dict[str, list[str]] = field(default_factory=dict)  # source -> tags
    unavailable: set[str] = field(default_factory=set)  # sources that errored, not "no match"


@dataclass
class ResolvedTrack:
    name: str
    spotify_id: str
    artists: list[ResolvedArtist]
    tags: dict[str, list[str]] = field(default_factory=dict)  # source -> tags
    unavailable: set[str] = field(default_factory=set)  # sources that errored, not "no match"


# --- LastFM helpers -----------------------------------------------------
#
# LastFM has no "look up artist by name" call worth trusting (too many
# collisions), so every LastFM lookup - here and in old/metadata.py before
# it - bounces through the artist's top track on Spotify: get the top track
# name, then lastfm.get_track(artist, title).get_artist(). This is also why
# a missing/empty top-track list is a legitimate "no match" outcome, not a
# bug - not every Spotify artist has enough listens for Spotify to surface a
# top track.
#
# get_artist()/get_top_tags()/get_similar()/get_top_tracks() below are plain
# pylast object methods, not LastFMClientProtocol methods - they make their
# own synchronous network calls, so each is still routed through
# asyncio.to_thread to keep the event loop unblocked, but (unlike
# lastfm.get_track itself) they don't get with_backoff retry: only the one
# Protocol-level call per client is worth retrying, chained lookups off of
# it are cheap to just report UNAVAILABLE on failure instead.


def _classify_lastfm_failure(exc: Exception):
    """WSError covers both 'rate-limited, exhausted lastfm.py's own retries'
    (still UNAVAILABLE - it never got a real answer) and 'well-formed error
    response, e.g. not found' (a confident miss, not UNAVAILABLE). Anything
    else (connection-level) is always UNAVAILABLE."""
    if isinstance(exc, pylast.WSError):
        return UNAVAILABLE if exc.get_id() == 29 else None
    return UNAVAILABLE


async def _sp_artist_top_track(spotify_id: str, spotify: SpotifyClientProtocol):
    try:
        top_tracks = await spotify.artist_top_tracks(spotify_id)
        return top_tracks["tracks"][0]["name"]
    except (KeyError, IndexError, TypeError):
        return None  # confident: artist has no top track data
    except Exception:
        return UNAVAILABLE  # Spotify itself errored


async def _sp_artist_to_lastfm(
    spotify_id: str,
    artist_name: str,
    spotify: SpotifyClientProtocol,
    lastfm: LastFMClientProtocol,
):
    """Returns a pylast Artist-like object, None on a confident no-match, or
    UNAVAILABLE if LastFM (or Spotify, for the top-track lookup) couldn't be
    reached/errored."""
    top_song = await _sp_artist_top_track(spotify_id, spotify)
    if top_song is UNAVAILABLE:
        return UNAVAILABLE
    if top_song is None:
        return None

    try:
        track = await lastfm.get_track(artist=artist_name, title=top_song)
        return await asyncio.to_thread(track.get_artist)
    except Exception as exc:
        return _classify_lastfm_failure(exc)


async def _lastfm_to_sp(lfm_artist, spotify: SpotifyClientProtocol) -> str | None:
    """Reverse of _sp_artist_to_lastfm: given a pylast Artist, find the matching
    Spotify artist id via the same top-track bounce. Used only by
    get_similar_artists, which treats any failure here as 'skip this one
    similar artist' rather than something worth tracking as UNAVAILABLE -
    it's a best-effort discovery list, not persisted metadata."""
    try:
        top_tracks = await asyncio.to_thread(lfm_artist.get_top_tracks)
        top_song = top_tracks[0].item.get_title()
        artist_name = await asyncio.to_thread(lfm_artist.get_name)
    except (IndexError, AttributeError):
        return None
    except Exception:
        return None

    try:
        res = await spotify.search(f"{artist_name} - {top_song}", type="track")
    except Exception:
        return None

    items = res.get("tracks", {}).get("items", [])
    if not items:
        return None

    for sp_artist in items[0]["artists"]:
        if artist_name.strip().lower() == sp_artist["name"].strip().lower():
            return sp_artist["id"]
    return None


async def _lastfm_top_tags(lfm_obj):
    """Works for both a pylast Artist and a pylast Track - both expose get_top_tags()."""
    try:
        top_tags = await asyncio.to_thread(lfm_obj.get_top_tags)
        return [tag.item.get_name().strip().capitalize() for tag in top_tags]
    except Exception as exc:
        result = _classify_lastfm_failure(exc)
        return result if result is UNAVAILABLE else []


# --- MusicBrainz helpers -------------------------------------------------
#
# musicbrainz.py already retries its own retryable errors (503s, network
# blips) internally before ever raising - so any exception that reaches this
# module means MusicBrainz genuinely failed to answer, never "not found"
# (that's just an empty recording-list, not an exception). Every except
# clause below is therefore unconditionally UNAVAILABLE, unlike the LastFM
# ones above.


async def _sp_artist_to_mb(
    spotify_id: str,
    artist_name: str,
    spotify: SpotifyClientProtocol,
    musicbrainz: MusicBrainzClientProtocol,
):
    """Finds the MusicBrainz artist id matching a Spotify artist, via the same
    top-track bounce as the LastFM lookup (MusicBrainz has no reliable
    search-artist-by-name-alone either)."""
    top_song = await _sp_artist_top_track(spotify_id, spotify)
    if top_song is UNAVAILABLE:
        return UNAVAILABLE
    if top_song is None:
        return None

    try:
        res = await musicbrainz.search_recordings(
            f'artist:"{artist_name}" AND recording:"{top_song}"'
        )
    except Exception:
        return UNAVAILABLE

    recordings = res.get("recording-list", [])
    if not recordings:
        return None

    for credit in recordings[0].get("artist-credit", []):
        if isinstance(credit, dict) and "artist" in credit:
            if artist_name.strip().lower() == credit["artist"]["name"].strip().lower():
                return credit["artist"]["id"]
    return None


async def _mb_artist_tags(musicbrainz_id: str | None, musicbrainz: MusicBrainzClientProtocol):
    if not musicbrainz_id:
        return []

    try:
        res = await musicbrainz.get_artist_by_id(musicbrainz_id, includes=["tags", "user-tags"])
        return [tag["name"].lower() for tag in res["artist"].get("tag-list", [])]
    except Exception:
        return UNAVAILABLE


async def _mb_track_tags(artist_name: str, track_name: str, musicbrainz: MusicBrainzClientProtocol):
    try:
        res = await musicbrainz.search_recordings(
            f'artist:"{artist_name}" AND recording:"{track_name}"'
        )
    except Exception:
        return UNAVAILABLE

    recordings = res.get("recording-list", [])
    if not recordings:
        return []

    try:
        tags_data = await musicbrainz.get_recording_by_id(
            recordings[0]["id"], includes=["tags", "user-tags"]
        )
        return [tag["name"] for tag in tags_data["recording"].get("tag-list", [])]
    except Exception:
        return UNAVAILABLE


# --- Public API ------------------------------------------------------------


async def resolve_artist(
    spotify_id: str,
    spotify: SpotifyClientProtocol,
    musicbrainz: MusicBrainzClientProtocol,
    lastfm: LastFMClientProtocol,
) -> ResolvedArtist:
    """Cross-reference a Spotify artist against MusicBrainz and LastFM, collecting tags
    from whichever sources produce a confident match. Never raises - a source with no
    match leaves its fields None/empty, a source that errored is added to
    `unavailable` instead."""
    sp_artist = await spotify.artist(spotify_id)
    name = sp_artist["name"]

    tags: dict[str, list[str]] = {}
    unavailable: set[str] = set()
    lastfm_name = None
    musicbrainz_id = None

    lfm_artist = await _sp_artist_to_lastfm(spotify_id, name, spotify, lastfm)
    if lfm_artist is UNAVAILABLE:
        unavailable.add("lastfm")
    elif lfm_artist is not None:
        lastfm_name = await asyncio.to_thread(lfm_artist.get_name)
        lastfm_tags = await _lastfm_top_tags(lfm_artist)
        if lastfm_tags is UNAVAILABLE:
            unavailable.add("lastfm")
        elif lastfm_tags:
            tags["lastfm"] = lastfm_tags

    mb_id = await _sp_artist_to_mb(spotify_id, name, spotify, musicbrainz)
    if mb_id is UNAVAILABLE:
        unavailable.add("musicbrainz")
    elif mb_id:
        musicbrainz_id = mb_id
        mb_tags = await _mb_artist_tags(musicbrainz_id, musicbrainz)
        if mb_tags is UNAVAILABLE:
            unavailable.add("musicbrainz")
        elif mb_tags:
            tags["musicbrainz"] = mb_tags

    return ResolvedArtist(
        name=name,
        spotify_id=spotify_id,
        musicbrainz_id=musicbrainz_id,
        lastfm_name=lastfm_name,
        tags=tags,
        unavailable=unavailable,
    )


async def resolve_track(
    spotify_id: str,
    spotify: SpotifyClientProtocol,
    musicbrainz: MusicBrainzClientProtocol,
    lastfm: LastFMClientProtocol,
) -> ResolvedTrack:
    """Cross-reference a Spotify track against MusicBrainz and LastFM.

    `artists` on the returned ResolvedTrack are name/spotify_id only, NOT fully
    cross-referenced - resolve_artist is deliberately not called here. Callers
    that need real ResolvedArtist data (musicbrainz_id, lastfm_name, tags) get
    it via resolve_artist/push_artist directly, once, on their own repo-miss
    path. Resolving every artist on every track here too would mean resolving
    a new artist twice (once for the track, once when services.py pushes it) -
    this only exists so ResolvedTrack.artists has something to enumerate ids
    from without a second Spotify round-trip."""
    track = await spotify.track(spotify_id)
    name = track["name"]

    artists = [
        ResolvedArtist(name=artist_data["name"], spotify_id=artist_data["id"])
        for artist_data in track["artists"]
    ]

    tags: dict[str, list[str]] = {}
    unavailable: set[str] = set()
    primary_artist_name = track["artists"][0]["name"] if track["artists"] else None

    if primary_artist_name:
        try:
            lfm_track = await lastfm.get_track(artist=primary_artist_name, title=name)
        except Exception as exc:
            lfm_track = None
            if _classify_lastfm_failure(exc) is UNAVAILABLE:
                unavailable.add("lastfm")

        if lfm_track is not None:
            lastfm_tags = await _lastfm_top_tags(lfm_track)
            if lastfm_tags is UNAVAILABLE:
                unavailable.add("lastfm")
            elif lastfm_tags:
                tags["lastfm"] = lastfm_tags

        mb_tags = await _mb_track_tags(primary_artist_name, name, musicbrainz)
        if mb_tags is UNAVAILABLE:
            unavailable.add("musicbrainz")
        elif mb_tags:
            tags["musicbrainz"] = mb_tags

    return ResolvedTrack(
        name=name, spotify_id=spotify_id, artists=artists, tags=tags, unavailable=unavailable
    )


async def get_similar_artists(
    spotify_artist_id: str,
    spotify: SpotifyClientProtocol,
    lastfm: LastFMClientProtocol,
    degrees: int = 1,
) -> list[str]:
    """Returns Spotify artist ids of artists similar to the given one, via LastFM's
    similarity graph, translated back to Spotify ids. A discovery list, not persisted
    metadata - LastFM being unavailable here just means an empty/short result, not
    something worth tracking for retry."""
    assert degrees > 0

    sp_artist = await spotify.artist(spotify_artist_id)
    lfm_artist = await _sp_artist_to_lastfm(spotify_artist_id, sp_artist["name"], spotify, lastfm)
    if lfm_artist is None or lfm_artist is UNAVAILABLE:
        return []

    try:
        similar = await asyncio.to_thread(lfm_artist.get_similar, limit=3)
    except Exception:
        return []

    similar_ids = []
    for sim in similar:
        sp_id = await _lastfm_to_sp(sim.item, spotify)
        if sp_id is not None:
            similar_ids.append(sp_id)

    if degrees == 1:
        return similar_ids

    nested = []
    for artist_id in similar_ids:
        nested.extend(await get_similar_artists(artist_id, spotify, lastfm, degrees - 1))

    # Dedupe while preserving discovery order (old code returned a list of
    # per-artist lists here, never flattened - fixed rather than ported as-is).
    return list(dict.fromkeys(similar_ids + nested))
