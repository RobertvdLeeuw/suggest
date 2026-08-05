"""
Strategies at the resolution.py domain level - ResolvedArtist/ResolvedTrack
and the tags/unavailable shapes they carry. Operate on the already-parsed
domain objects, not raw API JSON (that's strategies/apis.py's job) - these
feed resolution.py-level tests and mapping.py tests directly via
mocks/clients.py's fakes, not via mocks/raw_*.py.
"""

import hypothesis.strategies as st

from src.collecter.resolution import KNOWN_SOURCES, ResolvedArtist, ResolvedTrack
from strategies.apis import spotify_id_strat, unicode_name_strat

# tags_dict_strat: source -> list[str] dict, keys from KNOWN_SOURCES, values
# from unicode_name_strat (tag names hit the same unicode/empty-string edge
# cases artist names do).
tags_dict_strat = st.dictionaries(
    keys=st.sampled_from(sorted(KNOWN_SOURCES)),
    values=st.lists(unicode_name_strat, min_size=0, max_size=5),
)


@st.composite
def resolved_artist_strat(draw) -> ResolvedArtist:
    """A ResolvedArtist with random name, optional cross-service ids, and a
    tags/unavailable split that's disjoint by construction - draw
    `unavailable` first, then only ever draw tag keys from what's left of
    KNOWN_SOURCES, rather than drawing both independently and filtering: a
    source can't be both resolved and unavailable at once, that's the
    invariant resolution.py's docstring describes, and the strategy should
    never even be able to generate a state resolution.py itself would
    consider invalid."""
    unavailable = draw(st.sets(st.sampled_from(sorted(KNOWN_SOURCES))))
    available_sources = sorted(KNOWN_SOURCES - unavailable)

    tags = {}
    for source in available_sources:
        if draw(st.booleans()):
            tags[source] = draw(st.lists(unicode_name_strat, min_size=0, max_size=5))

    return ResolvedArtist(
        name=draw(unicode_name_strat),
        spotify_id=draw(st.one_of(st.none(), spotify_id_strat)),
        musicbrainz_id=draw(st.one_of(st.none(), st.uuids().map(str))),
        lastfm_name=draw(st.one_of(st.none(), unicode_name_strat)),
        tags=tags,
        unavailable=unavailable,
    )


@st.composite
def resolved_track_strat(draw) -> ResolvedTrack:
    """A ResolvedTrack with 0-few resolved_artist_strat() artists and the
    same disjoint tags/unavailable shape as resolved_artist_strat - see that
    function's docstring for why it's built this way rather than drawn
    independently."""
    unavailable = draw(st.sets(st.sampled_from(sorted(KNOWN_SOURCES))))
    available_sources = sorted(KNOWN_SOURCES - unavailable)

    tags = {}
    for source in available_sources:
        if draw(st.booleans()):
            tags[source] = draw(st.lists(unicode_name_strat, min_size=0, max_size=5))

    return ResolvedTrack(
        name=draw(unicode_name_strat),
        spotify_id=draw(spotify_id_strat),
        artists=draw(st.lists(resolved_artist_strat(), min_size=0, max_size=3)),
        tags=tags,
        unavailable=unavailable,
    )


# client_failure_sequence_strat: for resolution.py's resolve_artist/
# resolve_track property tests - draws a *trinary* outcome per client
# (spotify/musicbrainz/lastfm), not a binary fail/succeed. resolution.py's
# whole point is distinguishing "checked, found nothing" (a confident miss -
# permanent, never surfaces in `unavailable`) from "couldn't check at all"
# (transient - always surfaces in `unavailable`) - a binary model can't drive
# mocks.clients fakes into that distinction, which is exactly what the
# "unavailable vs confidently empty" assertions in unit/test_resolution.py
# need exercised.
#
# "success" here means the fake returns real (fixture-shaped) data; the
# actual ResolvedArtist/ResolvedTrack content produced from a "success" case
# isn't this strategy's concern - that's what strategies.apis's response
# strategies are for. This strategy only decides the failure/success axis
# each client fake is configured with for one resolution attempt.
CLIENT_NAMES = ("spotify", "musicbrainz", "lastfm")

client_failure_sequence_strat = st.fixed_dictionaries(
    {name: st.sampled_from(["success", "confident_miss", "unavailable"]) for name in CLIENT_NAMES}
)
