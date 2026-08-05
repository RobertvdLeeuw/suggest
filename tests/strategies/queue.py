"""
Strategies for SongQueue item generation, used by unit/test_song_queue.py's
stateful property test (see that file's module docstring) and by
integration/test_song_queue_concurrency.py's real-process tests.

Scope note (post-slimdown): the RuleBasedStateMachine in test_song_queue.py
now runs only against mocks.song_queue_double.SongQueueDouble, not the real
SongQueue class - see that file's docstring for why. This module only needs
to generate individual queue items, not operation sequences.
"""

import hypothesis.strategies as st

from strategies.apis import spotify_id_strat

# song_queue_item_strat: a single (filepath, spotify_id) tuple, matching what
# download.py actually .put()s onto a SongQueue (see _download_one/
# plan_downloads' queue_puts) - NOT a bare spotify_id or a Song/QueueObject
# ORM row, that was the old strategy's shape and doesn't match current
# SongQueue usage. filepath is loosely path-shaped rather than arbitrary
# text, since SongQueueDouble/SongQueue only ever compare items by equality
# (dedup-on-put) - the string doesn't need to be a real filesystem path for
# these tests, just consistent and hashable.
song_queue_item_strat = st.tuples(
    st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")), min_size=1, max_size=40).map(
        lambda name: f"/tmp/downloads/{name}.mp3"
    ),
    spotify_id_strat,
)

# queue_name_strat: small fixed set matching real SongQueue .name values used
# elsewhere (e.g. "jukemir"/"auditus") - not arbitrary text, these are the
# only two names __init__.py's EMBEDDERS registry actually constructs.
queue_name_strat = st.sampled_from(["jukemir", "auditus"])

# Dropped: operation_sequence_strat. RuleBasedStateMachine generates its own
# rule sequence via @rule decorators - a separate hand-rolled sequence
# strategy was redundant with that built-in machinery. Confirmed once
# test_song_queue.py was actually written; not needed.
