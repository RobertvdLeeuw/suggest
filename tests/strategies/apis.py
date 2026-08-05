"""
Strategies for raw API-shaped data - built as structural variations on the
golden templates in fixtures/api_responses.py, not generated from scratch.
See fixtures/api_responses.py's module docstring for why: these are real
captured Spotify/MusicBrainz response shapes, and inventing a schema with
st.text()/st.dictionaries() risks drifting from what the real APIs actually
return, plus risks quietly dropping the unicode/cross-artist edge cases
already baked into the golden data.

What's worth randomizing here is structure, not content: page counts, item
counts per page, presence/absence of optional fields, single vs multi-artist
tracks/albums, empty-result shapes. Name/text content should mostly come
from a small curated pool (ascii, unicode, empty string, very long) rather
than arbitrary generated text - see spotify_id_strat below for the one
exception, since spotify ids have a known fixed format worth generating
properly.
"""

import copy

import hypothesis.strategies as st
from fixtures.api_responses import (
    MB_ARTIST_BY_ID,
    MB_RECORDING_BY_ID,
    MB_SEARCH_RECORDINGS,
    SPOTIFY_ARTIST,
    SPOTIFY_CURRENT_PLAYBACK,
    SPOTIFY_SEARCH,
)
from mocks.raw_spotify import paginate

# unicode_name_strat: small curated pool of tricky name strings, rather than
# arbitrary st.text() - used wherever a strategy needs to substitute an
# artist/track name into a golden template. Kept as a plain sampled_from
# (not composite) since it's a leaf strategy other strategies compose with.
unicode_name_strat = st.sampled_from(
    [
        "Radiohead",
        "Mötley Crüe",
        "CAN",
        "Björk",
        "宇多田ヒカル",  # Utada Hikaru
        "Øystein",
        "Ke$ha",
        "",  # empty string - deliberately included, not filtered out
        "A" * 300,  # very long
        "  leading and trailing whitespace  ",
    ]
)

# spotify_id_strat: 22-char alphanumeric id, matching Spotify's real id format.
# (kept from the pre-refactor strategies/apis.py - this one was fine as-is.)
spotify_id_strat = st.from_regex(r"[A-Za-z0-9]{22}", fullmatch=True)


@st.composite
def spotify_artist_response_strat(draw) -> dict:
    """A structural variation on fixtures.SPOTIFY_ARTIST: swaps in a random
    name and toggles presence of the optional fields resolution.py/mapping.py
    actually read off an artist response (genres, images, followers) -
    doesn't invent fields the real endpoint doesn't return."""
    artist = copy.deepcopy(SPOTIFY_ARTIST)
    artist["name"] = draw(unicode_name_strat)

    if not draw(st.booleans()):
        artist["genres"] = []
    if not draw(st.booleans()):
        artist["images"] = []
    if draw(st.booleans()):
        artist.pop("followers", None)

    return artist


@st.composite
def spotify_paginated_response_strat(draw, template: dict, item_key: str) -> list[dict]:
    """Given one of the paginated golden templates (SPOTIFY_SAVED_TRACKS,
    SPOTIFY_ALBUM_TRACKS, SPOTIFY_ARTIST_ALBUMS, SPOTIFY_CURRENT_USER_PLAYLISTS)
    and the key its items live under, draws a page count (0/1/many) and an
    items-per-page count, and returns a ready-to-assign page sequence via
    mocks.raw_spotify.paginate().

    Deliberately flat: `template[item_key]` must already be the list to
    chunk. SPOTIFY_SEARCH's nested "tracks" shape does NOT go through this
    function directly - see spotify_search_paginated_strat below, which
    wraps this one rather than teaching it a nested-path lookup for a single
    caller.

    Item *content* isn't varied here - a real item drawn from the template is
    reused (cycled) to fill out however many items get requested, since this
    strategy's job is exercising the pagination loop's completeness, not
    item-shape variety (that's spotify_artist_response_strat's job, etc.).
    """
    real_items = template[item_key]
    assert real_items, f"template[{item_key!r}] must have at least one real item to draw from"

    n_pages = draw(st.integers(min_value=0, max_value=4))
    if n_pages == 0:
        return []

    items_per_page = draw(st.integers(min_value=1, max_value=3))
    total_items = n_pages * items_per_page
    items = [real_items[i % len(real_items)] for i in range(total_items)]

    return paginate(items, items_per_page)


@st.composite
def spotify_search_paginated_strat(draw) -> dict:
    """SPOTIFY_SEARCH's paginated part lives at ["tracks"], everything else
    in the response (the query echo, etc.) is static per page - so this
    draws pages via the generic strategy on template["tracks"] and re-nests
    each one under SPOTIFY_SEARCH's shape."""
    pages = draw(spotify_paginated_response_strat(SPOTIFY_SEARCH["tracks"], "items"))
    if not pages:
        return {**SPOTIFY_SEARCH, "tracks": {**SPOTIFY_SEARCH["tracks"], "items": [], "next": None}}
    return [{**SPOTIFY_SEARCH, "tracks": page} for page in pages]


@st.composite
def spotify_playback_snapshot_strat(draw) -> dict | None:
    """A single current_playback()-shaped dict (or None, matching the real
    endpoint's "nothing playing" response) - structural variation on
    SPOTIFY_CURRENT_PLAYBACK (is_playing True/False, item present/absent,
    item.type track/episode/None).

    NOTE: also used by strategies/listen.py for process_playback_tick's
    sequence generators - those wrap this one rather than defining their own
    snapshot shape, so there's one idea of what a valid snapshot looks like.
    """
    if draw(st.booleans()):
        return None

    snapshot = copy.deepcopy(SPOTIFY_CURRENT_PLAYBACK)
    is_playing = draw(st.booleans())
    snapshot["is_playing"] = is_playing

    item_type = draw(st.sampled_from(["track", "episode", None]))
    if item_type is None:
        snapshot["item"] = None
    else:
        snapshot["item"]["type"] = item_type
        snapshot["item"]["id"] = draw(spotify_id_strat)
        snapshot["item"]["duration_ms"] = draw(st.integers(min_value=1_000, max_value=600_000))
        snapshot["progress_ms"] = draw(
            st.integers(min_value=0, max_value=snapshot["item"]["duration_ms"])
        )

    return snapshot


@st.composite
def musicbrainz_search_recordings_strat(draw) -> dict:
    """Structural variation on MB_SEARCH_RECORDINGS: empty recording-list, or
    the real recording-list with artist-credit aliases present/absent."""
    result = copy.deepcopy(MB_SEARCH_RECORDINGS)

    if draw(st.booleans()):
        result["recording-list"] = []
        return result

    for recording in result.get("recording-list", []):
        for credit in recording.get("artist-credit", []):
            if isinstance(credit, dict) and "artist" in credit:
                if not draw(st.booleans()):
                    credit["artist"].pop("alias-list", None)

    return result


@st.composite
def musicbrainz_artist_by_id_strat(draw) -> dict:
    """Structural variation on MB_ARTIST_BY_ID: tag-list absent (the golden
    fixture's own shape - a real, legitimate response has no tags), present
    with a couple of tag names, or present-but-empty. _mb_artist_tags reads
    this via `.get("tag-list", [])`, so absent and empty must both resolve
    to `[]` downstream - exercising both is the point, not just one of
    them."""
    result = copy.deepcopy(MB_ARTIST_BY_ID)

    variant = draw(st.sampled_from(["absent", "empty", "present"]))
    if variant == "empty":
        result["artist"]["tag-list"] = []
    elif variant == "present":
        names = draw(st.lists(unicode_name_strat, min_size=1, max_size=3))
        result["artist"]["tag-list"] = [{"name": n, "count": "1"} for n in names]
    # "absent": leave the fixture's own shape (no tag-list key) untouched.

    return result


@st.composite
def musicbrainz_recording_by_id_strat(draw) -> dict:
    """Structural variation on MB_RECORDING_BY_ID: tag-list absent/empty/present -
    same rationale as musicbrainz_artist_by_id_strat above, mirrored for
    _mb_track_tags's `tags_data["recording"].get("tag-list", [])` read."""
    result = copy.deepcopy(MB_RECORDING_BY_ID)

    variant = draw(st.sampled_from(["absent", "empty", "present"]))
    if variant == "empty":
        result["recording"]["tag-list"] = []
    elif variant == "present":
        names = draw(st.lists(unicode_name_strat, min_size=1, max_size=3))
        result["recording"]["tag-list"] = [{"name": n, "count": "1"} for n in names]

    return result
