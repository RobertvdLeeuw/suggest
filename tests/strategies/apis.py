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

# spotify_id_strat: 22-char alphanumeric id, matching Spotify's real id format.
# (kept from the pre-refactor strategies/apis.py - this one was fine as-is.)
# touches: hypothesis.strategies.from_regex

# spotify_artist_response_strat: draws a page-count/optional-field variation on
# fixtures.api_responses.SPOTIFY_ARTIST.
# touches: fixtures.api_responses.SPOTIFY_ARTIST

# spotify_paginated_response_strat(template, item_key): generic paginator -
# given one of the paginated golden templates (SPOTIFY_SAVED_TRACKS,
# SPOTIFY_ALBUM_TRACKS, SPOTIFY_ARTIST_ALBUMS, SPOTIFY_CURRENT_USER_PLAYLISTS,
# SPOTIFY_SEARCH's nested "tracks"), draws a random number of pages
# (0/1/many) and items per page, with correctly-chained 'next' urls (or None
# on the last page) - this is what actually exercises clients/spotify.py's
# pagination-completeness loop, not any property of the item content itself.
# touches: fixtures.api_responses (the *_SAVED_TRACKS/*_ALBUM_TRACKS/etc constants)

# spotify_playback_snapshot_strat: single current_playback()-shaped dict,
# structural variation (is_playing True/False, item present/absent, type
# track/episode/None) on SPOTIFY_CURRENT_PLAYBACK - NOTE this is also needed
# by strategies/listen.py for process_playback_tick; decide whether
# strategies/listen.py imports this or defines its own - probably imports
# this one and adds the *sequence* generation on top, to avoid two divergent
# ideas of what a valid snapshot looks like.
# touches: fixtures.api_responses.SPOTIFY_CURRENT_PLAYBACK

# musicbrainz_search_recordings_strat / musicbrainz_artist_by_id_strat /
# musicbrainz_recording_by_id_strat: structural variations (empty
# recording-list, missing optional fields, alias-list present/absent) on
# fixtures.api_responses.MB_*.
# touches: fixtures.api_responses.MB_SEARCH_RECORDINGS, MB_ARTIST_BY_ID, MB_RECORDING_BY_ID

# unicode_name_strat: small curated pool of tricky name strings (plain ascii,
# accented latin, CJK, empty string, very long) rather than arbitrary
# st.text() - used wherever a strategy needs to substitute an artist/track
# name into a golden template.
# touches: hypothesis.strategies.sampled_from
