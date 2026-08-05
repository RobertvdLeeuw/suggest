# Was empty pre-refactor too - nothing to carry over.
#
# Scope note: nothing in the current unit/integration test plan exercises
# embedders/jukemir.py or embedders/auditus.py's actual embed functions
# directly (see embedders/song_queue.py's JukeMIREmbedFunc/AuditusEmbedFunc
# Protocols) - the plan only covers SongQueue's queueing behavior and the
# embedders/__init__.py process-orchestration layer via
# integration/test_embedding_pipeline.py, treating the embed functions
# themselves as opaque. If we want real/fake audio file strategies later
# (for testing jukemir.py/auditus.py's embed() functions directly), they'd
# go here - fake waveform generation, or a small real-audio-file fixture set
# a-la fixtures/api_responses.py. Flagging as a gap rather than guessing at
# a shape we haven't scoped yet.
