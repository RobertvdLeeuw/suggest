"""
Strategies for SongQueue operation sequences, used by unit/test_song_queue.py's
stateful property tests (see that file's module docstring) and by
integration/test_song_queue_concurrency.py's real-process stress test.

The pre-refactor version of this file imported collecter.downloader's
start_download_loop/clean_downloads/_download and collecter.embedders'
start_processes directly - none of those names exist anymore post-refactor
(split across download.py/download_tracking.py/embedders/__init__.py now).
Rebuilt clean rather than patched.
"""

# song_queue_item_strat: a single (filepath, spotify_id) tuple, matching what
# download.py actually .put()s onto a SongQueue (see _download_one/plan_downloads'
# queue_puts) - NOT a bare spotify_id or a Song/QueueObject ORM row, that was
# the old strategy's shape and doesn't match current SongQueue usage.
# touches: strategies.apis.spotify_id_strat

# operation_sequence_strat: a list of operations (each one of
# put(item)/get()/remove(item)/peek()/peek_all()) for
# unit/test_song_queue.py's RuleBasedStateMachine to replay - only needed if
# the state machine doesn't generate its own rule sequence via hypothesis's
# built-in stateful machinery (RuleBasedStateMachine normally handles
# sequencing itself via @rule decorators, so this may not be needed at all -
# confirm once test_song_queue.py's actual structure is written before
# building this).

# queue_name_strat: small fixed set matching real SongQueue .name values used
# elsewhere (e.g. "jukemir"/"auditus") - kept here rather than duplicated in
# strategies/download.py's wanted_by_queue_strat; that one should import this.
